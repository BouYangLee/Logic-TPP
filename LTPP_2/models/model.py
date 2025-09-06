import torch
import torch.nn as nn
import torch.nn.functional as F
from .softlogic import SoftLogicEngine
from constants import total_predicate_set, mental_predicate_set, action_predicate_set, grid_length
import math

BEFORE = "BEFORE"
AFTER  = "AFTER"
EQUAL  = "EQUAL"

class Rule:
    def __init__(self, head, body_predicate_idx, rule_type, name=None,
                 temporal_pairs=None, temporal_types=None):
        """
        head: int (predicate id)
        body_predicate_idx: List[int]  (只放普通谓词 id！不要放 "AFTER(2,4)" 这样的字符串)
        rule_type: 'A->M' | 'M->M' | 'M->A'
        temporal_pairs: List[(a,b)]  —— 每个元素是 (int a, int b)
        temporal_types: List[str]    —— 对应 BEFORE/AFTER/EQUAL
        """
        self.head = int(head)
        self.body = [int(x) for x in body_predicate_idx]
        self.body_predicate_idx = list(self.body)
        self.type = rule_type
        self.name = name if name is not None else f"r{self.head}_{'_'.join(map(str, self.body))}"

        self.temporal_pairs = []
        self.temporal_types = []
        if temporal_pairs is not None:
            if temporal_types is None or len(temporal_pairs) != len(temporal_types):
                raise ValueError("temporal_pairs and temporal_types must have the same length")
            for (a, b), t in zip(temporal_pairs, temporal_types):
                self.temporal_pairs.append((int(a), int(b)))
                assert t in (BEFORE, AFTER, EQUAL), f"Unknown temporal type {t}"
                self.temporal_types.append(t)

class LA_TPP_Model(nn.Module):
    def __init__(self, rules, mental_predicates, action_predicates,
                 predicate_list=total_predicate_set, d_pred=6, device='cpu',
                 learnable_K=False):
        super().__init__()
        self.device = torch.device(device if isinstance(device, str) else device)

        # 先合并常量集合 + 规则里出现的所有谓词 id（只合并数字 id）
        base_preds = set(int(p) for p in predicate_list)
        mentals = set(int(p) for p in mental_predicates)
        actions = set(int(p) for p in action_predicates)
        for r in rules:
            base_preds.add(int(r.head))
            for b in r.body:
                base_preds.add(int(b))
            # 根据 rule_type，自动把 head 放进对应集合
            if r.type in ('A->M', 'M->M'):
                mentals.add(int(r.head))
            elif r.type == 'M->A':
                actions.add(int(r.head))

        self.predicate_list = sorted(base_preds)
        self.pred_to_pos = {p: i for i, p in enumerate(self.predicate_list)}
        self.mental_predicates = sorted(mentals)
        self.action_predicates = sorted(actions)

        # 保存规则
        self.rules = rules

        # clause keys
        self.clause_keys = []
        self.clause_key_to_rule = {}
        self.key_to_str = {}
        used_names = set()
        for idx, r in enumerate(rules):
            key = (r.head, idx)
            self.clause_keys.append(key)
            self.clause_key_to_rule[key] = r
            base_name = r.name
            name = base_name
            suffix = 0
            while name in used_names:
                suffix += 1
                name = f"{base_name}_{suffix}"
            used_names.add(name)
            self.key_to_str[key] = name

        # embedding K
        self.d = d_pred
        K0 = torch.randn(len(self.predicate_list), self.d) * 0.1
        self.K = nn.Parameter(K0, requires_grad=bool(learnable_K))

        # rule thetas（保留：如果将来切回 soft-matching 用得到）
        self.Thetas = nn.ParameterDict()
        for key in self.clause_keys:
            rule = self.clause_key_to_rule[key]
            h = len(rule.body)
            param_name = self.key_to_str[key]
            theta_init = torch.randn(self.d, h) * 0.1
            for slot_idx, pred_id in enumerate(rule.body):
                if pred_id in self.pred_to_pos:
                    pred_pos = self.pred_to_pos[pred_id]
                    theta_init[:, slot_idx] += 0.3 * K0[pred_pos, :] + torch.randn(self.d) * 0.05
            self.Thetas[param_name] = nn.Parameter(theta_init)

        self.mental_to_idx = {p: i for i, p in enumerate(self.mental_predicates)}
        self.alpha_un = nn.Parameter(torch.ones(len(self.mental_predicates)) * (-0.5))

        # gamma 参数（A->M 用于 S_j，M->A 用于 λ^logic）
        self.gamma_AtoM = nn.ParameterDict()
        self.gamma_MtoA = nn.ParameterDict()
        for key in self.clause_keys:
            ks = self.key_to_str[key]
            self.gamma_AtoM[ks] = nn.Parameter(torch.zeros(len(self.mental_predicates)))
            self.gamma_MtoA[ks] = nn.Parameter(torch.zeros(len(self.action_predicates)))

            rule = self.clause_key_to_rule[key]
            if rule.type in ('A->M', 'M->M'):  # 头是 mental
                j_idx = self.mental_predicates.index(rule.head)
                with torch.no_grad():
                    self.gamma_AtoM[ks].zero_()
                    self.gamma_AtoM[ks].data[j_idx] = 0.2
            if rule.type == 'M->A':           # 头是 action
                k_idx = self.action_predicates.index(rule.head)
                with torch.no_grad():
                    self.gamma_MtoA[ks].zero_()
                    self.gamma_MtoA[ks].data[k_idx] = 0.2

        self.b = nn.Parameter(torch.ones(len(self.action_predicates)) * (-2.0))
        self.w = nn.Parameter(0.2 * torch.randn(len(self.action_predicates), len(self.mental_predicates)))

        # exact 推理索引：包含“所有 head”（mental + action）
        self.clauses_by_head_exact_all = {}
        for key in self.clause_keys:
            rule = self.clause_key_to_rule[key]
            self.clauses_by_head_exact_all.setdefault(rule.head, []).append(
                (key, list(rule.body), rule)
            )

        self.engine = SoftLogicEngine(
            predicate_list=self.predicate_list,
            d=self.d,
            K_init=None,
            T_match=0.6,
            tau=0.1,
            beta=5.0
        )
        self.engine.K = self.K
        self.to(self.device)

        self.last_debug = {}

    def get_theta(self, key):
        param_name = self.key_to_str[key]
        return self.Thetas[param_name]

    def alpha(self):
        return F.softplus(self.alpha_un)

    def build_val_for_microstep(self, actions_rank0, M_minus):
        """
        把 rank=0 的动作标 1，再把 M_minus 拷进 val。
        """
        P = len(self.predicate_list)
        val = torch.zeros(P, device=M_minus.device)
        for j, pid in enumerate(self.mental_predicates):
            pos = self.pred_to_pos[pid]
            val[pos] = torch.clamp(M_minus[j], 0.0, 1.0)
        for a in actions_rank0:
            if a in self.pred_to_pos:
                val[self.pred_to_pos[a]] = 1.0
        return val

    # === ODE 相关 ===
    def _compute_S_from_gcache(self, g_cache):
        """
        S_j(t)：只来自 head 是 mental 的规则（A->M or M->M）
        """
        m = len(self.mental_predicates)
        S = torch.zeros(m, device=self.K.device)
        for key, g in g_cache.items():
            rule = self.clause_key_to_rule[key]
            if rule.type in ('A->M', 'M->M'):  # head 是 mental
                ks = self.key_to_str[key]
                gamma = self.gamma_AtoM[ks]
                if isinstance(g, (int, float)):
                    g = torch.tensor(g, device=self.K.device)
                S = S + gamma * g
        return F.relu(S)

    def _compute_intensity_from_gcache(self, M_vec, g_cache):
        """
        λ_k(t) = softplus(b_k + w_k^T M) + sum_{M->A rules} γ_{k,f} g_f
        """
        linear = self.b + (self.w @ M_vec)
        lam_ment = F.softplus(linear)

        lam_logic = torch.zeros_like(lam_ment)
        for key, g in g_cache.items():
            rule = self.clause_key_to_rule[key]
            if rule.type == 'M->A':  # head 是 action
                ks = self.key_to_str[key]
                gamma_vec = self.gamma_MtoA[ks]
                if isinstance(g, (int, float)):
                    g = torch.tensor(g, device=self.K.device)
                lam_logic = lam_logic + gamma_vec * g
        lam_logic = F.relu(lam_logic)
        return lam_ment + lam_logic, lam_ment, lam_logic

    # === 时序门控 ===
    def _temporal_gate(self, rule, rank_map):
        """
        所有时序对都满足 -> 1；否则 0。
        当某一端未出现（无 rank）时，判 0。
        """
        if not rule.temporal_pairs:
            return torch.tensor(1.0, device=self.K.device)
        for (a, b), rel in zip(rule.temporal_pairs, rule.temporal_types):
            ra = rank_map.get(int(a), None)
            rb = rank_map.get(int(b), None)
            if ra is None or rb is None:
                return torch.tensor(0.0, device=self.K.device)
            if rel == BEFORE and not (ra < rb): return torch.tensor(0.0, device=self.K.device)
            if rel == AFTER  and not (ra > rb): return torch.tensor(0.0, device=self.K.device)
            if rel == EQUAL  and not (ra == rb):return torch.tensor(0.0, device=self.K.device)
        return torch.tensor(1.0, device=self.K.device)

    # === 微步推理（带时序） ===
    def exact_reason_microstep_temporal(self, actions_at_t, scheduled_actions_at_t, M_minus,
                                        g_threshold=0.0, delta_threshold=1e-6, debug=False):
        """
        把 (外生动作 ∪ 合并调度动作) 作为 rank=0；迭代激活 mental；M->A 不落地，调度到下一刻。
        返回：
          M_plus, val_post, g_cache, trace, inferred_actions_this_step
        """
        rank0_actions = list(sorted(set(list(actions_at_t) + list(scheduled_actions_at_t))))
        val = self.build_val_for_microstep(rank0_actions, M_minus)
        g_cache = {}
        trace = []

        # rank：动作 rank=0；每轮新激活的 mental 在收尾时记 rank=it+1
        rank_map = {int(a): 0 for a in rank0_actions}

        max_iters = 8
        for it in range(max_iters):
            changed = False
            val_new = val.clone()
            newly_activated_mentals = set()
            inferred_actions = set()

            # 遍历“所有 head”的精确规则
            for head, clause_list in self.clauses_by_head_exact_all.items():
                g_list = []
                infos = []
                for key, body_ids, rule in clause_list:
                    # 体值（不含时序）
                    g_body = self.engine.eval_clause_exact(body_ids, val, tau=0.1)
                    # 时序门控
                    gate = self._temporal_gate(rule, rank_map)
                    g = g_body * gate
                    g_list.append(g)
                    infos.append((key, g, body_ids, rule))

                if len(g_list) == 0:
                    continue

                beta = 5.0
                g_stack = torch.stack(g_list)
                g_head = (1.0 / beta) * torch.logsumexp(beta * g_stack, dim=0)

                old_val = 0.0
                if head in self.pred_to_pos:
                    old_val = float(val[self.pred_to_pos[head]].item())

                new_val = max(old_val, float(g_head.item()))
                delta = new_val - old_val

                # 记录所有子规则的 g
                for key, g, body_ids, rule in infos:
                    g_cache[key] = float(g.item())

                if head in self.mental_predicates:
                    if delta > delta_threshold:
                        pos = self.pred_to_pos[head]
                        val_new[pos] = new_val
                        changed = True
                        newly_activated_mentals.add(int(head))
                        # 追踪（用于可视化）
                        max_idx = int(torch.argmax(g_stack).item())
                        key_chosen, g_chosen, body_ids, rule = infos[max_idx]
                        trace.append({
                            'iter': int(it),
                            'clause_key': key_chosen,
                            'head': int(head),
                            'g': float(g_chosen.item()),
                            'matched_predicates': list(body_ids),
                            'prev_val': old_val,
                            'new_val': new_val,
                            'delta': delta
                        })

                elif head in self.action_predicates:
                    # 本刻不落地：只要 g_head 足够，推到下一刻
                    if float(g_head.item()) > g_threshold:
                        inferred_actions.add(int(head))
                        # 追踪（动作头也记一下，便于 debug）
                        max_idx = int(torch.argmax(g_stack).item())
                        key_chosen, g_chosen, body_ids, rule = infos[max_idx]
                        trace.append({
                            'iter': int(it),
                            'clause_key': key_chosen,
                            'head': int(head),
                            'g': float(g_chosen.item()),
                            'matched_predicates': list(body_ids),
                            'prev_val': 0.0,
                            'new_val': float(g_head.item()),
                            'delta': float(g_head.item())
                        })

            # 一轮结束：把新 mental 记 rank
            for h in newly_activated_mentals:
                rank_map[int(h)] = it + 1

            val = val_new
            if not changed:
                # 没有新的 mental 再出现就停；动作头的推理结果我们已收集到 inferred_actions
                inferred_actions_this_step = list(sorted(inferred_actions))
                # M_plus：从 val 里读 mental 分量
                M_plus = torch.zeros(len(self.mental_predicates), device=self.K.device)
                for j, pid in enumerate(self.mental_predicates):
                    M_plus[j] = torch.clamp(val[self.pred_to_pos[pid]], 0.0, 1.0)
                return M_plus, val, g_cache, trace, inferred_actions_this_step

        # 达到最大迭代也返回
        inferred_actions_this_step = []
        M_plus = torch.zeros(len(self.mental_predicates), device=self.K.device)
        for j, pid in enumerate(self.mental_predicates):
            M_plus[j] = torch.clamp(val[self.pred_to_pos[pid]], 0.0, 1.0)
        return M_plus, val, g_cache, trace, inferred_actions_this_step

    # === 主循环 ===
    def handle_event_sequence(self, events, recent_window=None, return_trace=False,
                              g_threshold=0.05, delta_threshold=1e-4, debug_verbose=False):
        """
        events: List[(action_id:int, time:float)]
        """
        device = self.K.device
        if recent_window is None:
            recent_window = grid_length

        # 分组：同一时刻的外生动作
        time_groups = {}
        for (a, t) in events:
            time_groups.setdefault(float(t), []).append(int(a))
        sorted_times = sorted(time_groups.keys())

        # 调度表：把本刻 M->A 推理出来的动作放到下一刻
        schedule = {}

        total_loglik = torch.tensor(0.0, device=device)
        total_survival = torch.tensor(0.0, device=device)
        event_trace_list = []
        t_prev = 0.0
        M = torch.zeros(len(self.mental_predicates), device=device)

        for idx, t_i in enumerate(sorted_times):
            actions_at_t = time_groups[t_i]
            delta_t = t_i - t_prev

            # 合并上一刻调度的动作
            scheduled_here = list(sorted(set(schedule.pop(t_i, set()))))

            # Step 1: ODE 从 t_{i-1}^+ 推到 t_i^-
            if delta_t > 0:
                # 先用当前 M 构一个 val（只有 mental），过一次 A->M 以便得到 S（稳定做法）
                val_for_ode = torch.zeros(len(self.predicate_list), device=device)
                for j, p in enumerate(self.mental_predicates):
                    val_for_ode[self.pred_to_pos[p]] = M[j]
                # 不需要 trace；只为了 S
                # 这里使用 g_cache 只来源 A->M/M->M（在 _compute_S_from_gcache 里已过滤）
                g_for_ode = {}
                S_current = self._compute_S_from_gcache(g_for_ode)
                alpha_vec = self.alpha()
                M_new = M + delta_t * (-alpha_vec * M + (1.0 - M) * S_current)
                M_minus = torch.clamp(M_new, 0.0, 1.0)
            else:
                M_minus = M.clone()

            # Step 2: 计算 λ(t_i^-) 并积累似然
            # 这里我们用当前 M_minus 计算 “直接驱动 + 规则 boost”
            # 先算 g_cache（只要 A->M/M->M 就够了，M->A 不参与 M 的 ODE）
            g_pre = {}
            lambda_pre, _, _ = self._compute_intensity_from_gcache(M_minus, g_pre)

            for a in actions_at_t:
                if a in self.action_predicates:
                    k_idx = self.action_predicates.index(a)
                    lam = torch.clamp(lambda_pre[k_idx], min=1e-8)
                    total_loglik = total_loglik + torch.log(lam)

            total_survival = total_survival + delta_t * torch.sum(lambda_pre)

            # Step 3: 微步推理（把外生动作 + 合并调度动作都当 rank=0）
            M_plus, val_post, g_post, trace, inferred_actions = self.exact_reason_microstep_temporal(
                actions_at_t, scheduled_here, M_minus,
                g_threshold=g_threshold, delta_threshold=delta_threshold, debug=debug_verbose
            )

            # Step 4: 更新 M（微步之后做一次“瞬时跳”）
            S_post = self._compute_S_from_gcache(g_post)
            M = torch.clamp(M_minus + (1.0 - M_minus) * S_post, 0.0, 1.0)

            # Step 5: 把本刻推理出来的动作调度到下一刻
            next_t = sorted_times[idx + 1] if idx + 1 < len(sorted_times) else (t_i + grid_length)
            if inferred_actions:
                schedule.setdefault(next_t, set()).update(inferred_actions)

            # 记录 trace（也把 scheduled_here 和 inferred_actions_this_step 带上）
            event_trace_list.append({
                'time': float(t_i),
                'actions': list(sorted(set(actions_at_t))),
                'scheduled_merged': list(sorted(set(scheduled_here))),
                'inferred_actions_for_next_time': list(sorted(set(inferred_actions))),
                'mental_states_after': M.clone(),
                'trace': trace
            })

            t_prev = t_i

        # 全局链（把“本刻外生动作 + 合并调度来本刻的动作”都作为 action 打印）
        global_chain = []
        for ev in event_trace_list:
            t = ev['time']
            for a in ev.get('actions', []):
                global_chain.append(f"action:{a}@{t:.3f}")
            for a in ev.get('scheduled_merged', []):
                global_chain.append(f"action:{a}@{t:.3f}")
            # mental 头：按照 iter 顺序，去重
            trace = ev.get('trace', [])
            rs = []
            for inf in trace:
                head = int(inf.get('head', -1))
                if head in self.mental_predicates:
                    rs.append((int(inf.get('iter', 0)), head))
            rs.sort(key=lambda x: x[0])
            seen = set()
            for _, h in rs:
                if h in seen: continue
                seen.add(h)
                global_chain.append(f"mental:{h}@{t:.3f}")

        self.last_debug = {'global_chain': global_chain, 'event_trace_list': event_trace_list}

        if return_trace:
            return total_loglik, total_survival, event_trace_list, global_chain
        else:
            return total_loglik, total_survival
