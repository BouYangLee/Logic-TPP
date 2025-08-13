import torch
import torch.nn as nn
import torch.nn.functional as F
from .softlogic import SoftLogicEngine
from constants import total_predicate_set, mental_predicate_set, action_predicate_set, grid_length

class Rule:
    def __init__(self, head, body_predicate_idx, rule_type, name=None):
        self.head = head
        self.body = list(body_predicate_idx)
        self.type = rule_type
        self.name = name if name is not None else f"r{self.head}_{'_'.join(map(str,self.body))}"

class LA_TPP_Model(nn.Module):
    def __init__(self, rules, mental_predicates, action_predicates,
                 predicate_list=total_predicate_set, d_pred=6, device='cpu',
                 learnable_K=False):
        super().__init__()
        self.device = device
        self.rules = rules
        self.predicate_list = list(predicate_list)
        self.pred_to_pos = {p: i for i, p in enumerate(self.predicate_list)}
        self.mental_predicates = mental_predicates
        self.action_predicates = action_predicates

        # --- build clause keys robustly and unique string names ---
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

        # embedding sizes
        self.d = d_pred
        K0 = torch.randn(len(self.predicate_list), self.d) * 0.1
        self.K = nn.Parameter(K0, requires_grad=bool(learnable_K))

        # per-clause Theta stored as 2D parameters (d, h)
        self.Thetas = nn.ParameterDict()
        for key in self.clause_keys:
            rule = self.clause_key_to_rule[key]
            h = len(rule.body)
            param_name = self.key_to_str[key]
            self.Thetas[param_name] = nn.Parameter(0.05 * torch.randn(self.d, h))

        # mental state indexing
        self.mental_to_idx = {p: i for i, p in enumerate(self.mental_predicates)}

        # dynamics and intensity params
        self.alpha_un = nn.Parameter(torch.randn(len(self.mental_predicates)) - 1.0)
        self.gamma_AtoM = nn.ParameterDict()
        self.gamma_MtoA = nn.ParameterDict()
        for key in self.clause_keys:
            ks = self.key_to_str[key]
            self.gamma_AtoM[ks] = nn.Parameter(torch.zeros(len(self.mental_predicates)))
            self.gamma_MtoA[ks] = nn.Parameter(torch.zeros(len(self.action_predicates)))
            rule = self.clause_key_to_rule[key]
            if rule.head in self.mental_predicates:
                j_idx = self.mental_predicates.index(rule.head)
                with torch.no_grad():
                    self.gamma_AtoM[ks].zero_()
                    self.gamma_AtoM[ks].data[j_idx] = 0.05
            if rule.head in self.action_predicates:
                k_idx = self.action_predicates.index(rule.head)
                with torch.no_grad():
                    self.gamma_MtoA[ks].zero_()
                    self.gamma_MtoA[ks].data[k_idx] = 0.05

        self.b = nn.Parameter(torch.zeros(len(self.action_predicates)))
        self.w = nn.Parameter(0.1 * torch.randn(len(self.action_predicates), len(self.mental_predicates)))

        # engine: use shared K reference
        self.engine = SoftLogicEngine(predicate_list=self.predicate_list, d=self.d, K_init=None)
        # share K tensor so softmatching uses same K
        self.engine.K = self.K
        self.to(device)

    def get_theta(self, key):
        param_name = self.key_to_str[key]
        return self.Thetas[param_name]

    def build_val_from_history(self, H_t_events, M_vec, t_now, recent_window=None):
        """
        Build the predicate truth-value vector given:
         - H_t_events: list of (action_pred, time)
         - M_vec: mental state vector (tensor) aligned with self.mental_predicates
         - t_now: reference time to decide which recent actions are active
        """
        if recent_window is None:
            recent_window = grid_length
        val = torch.zeros(len(self.predicate_list), device=self.K.device)
        # fill mental states
        for p in self.mental_predicates:
            if p in self.pred_to_pos:
                val[self.pred_to_pos[p]] = M_vec[self.mental_to_idx[p]]
        # mark recent actions as true
        for (pred, ts) in H_t_events:
            if pred in self.pred_to_pos and pred in self.action_predicates:
                if (t_now - ts) <= recent_window + 1e-9:
                    val[self.pred_to_pos[pred]] = 1.0
        return val

    def compute_S(self, g_cache):
        """
        Compute S vector for A->M contributions: sum_k gamma_AtoM[k] * g_k
        """
        m = len(self.mental_predicates)
        S = torch.zeros(m, device=self.K.device)
        for key, g in g_cache.items():
            ks = self.key_to_str[key]
            gamma = self.gamma_AtoM[ks]
            # g is a scalar tensor
            S = S + gamma * g
        return S

    def alpha(self):
        return F.softplus(self.alpha_un)

    def compute_intensity_vectors(self, M_vec, val_vec, g_cache):
        """
        Compute intensities: base lambda_ment from M_vec and lambda_logic from rule g's (M->A)
        """
        linear = self.b + (self.w @ M_vec)
        lambda_ment = F.softplus(linear)
        lambda_logic = torch.zeros_like(lambda_ment)
        for key, g in g_cache.items():
            ks = self.key_to_str[key]
            gamma_vec = self.gamma_MtoA[ks]
            lambda_logic = lambda_logic + gamma_vec * g
        return lambda_ment + lambda_logic, lambda_ment, lambda_logic

    def forward_chaining_all(self, val_init, M_vec, max_iters=4, return_trace=False):
        Thetas = {}
        for key in self.clause_keys:
            Thetas[key] = self.get_theta(key)
        # engine.forward_chaining_with_trace returns (val_out, g_cache, trace)
        val_out, g_cache, trace = self.engine.forward_chaining_with_trace(Thetas, val_init, max_iters=max_iters)
        if return_trace:
            return val_out, g_cache, trace
        else:
            return val_out, g_cache

    def handle_event_sequence(self, events, recent_window=None, return_trace=False, g_threshold=0.05, delta_threshold=1e-4):
        """
        Process sequence of (action_id, time). Returns loglik, survival and optionally traces & global_chain.
        """
        if recent_window is None:
            recent_window = grid_length
        device = self.K.device
        M = torch.zeros(len(self.mental_predicates), device=device)
        history = []
        t_prev = 0.0
        total_loglik = torch.tensor(0.0, device=device)
        total_survival = torch.tensor(0.0, device=device)

        event_trace_list = []

        for (action_id, t_i) in events:
            delta_t = t_i - t_prev
            # forward from previous time to t_prev -> val_pre uses M at t_prev
            val_pre = self.build_val_from_history(history, M, t_prev, recent_window)
            val_pre_fc, g_pre = self.forward_chaining_all(val_pre, M)
            S_prev = self.compute_S(g_pre)
            alpha_vec = self.alpha()

            # integrate M from t_prev to t_i (explicit Euler)
            M_minus = M + delta_t * (-alpha_vec * M + (1.0 - M) * S_prev)
            M_minus = torch.clamp(M_minus, 0.0, 1.0)

            # values immediately before event (small epsilon before t_i)
            val_pre_at_ti = self.build_val_from_history(history, M_minus, t_i - 1e-9, recent_window)
            val_fc_before, g_before = self.forward_chaining_all(val_pre_at_ti, M_minus)
            lambda_k, _, _ = self.compute_intensity_vectors(M_minus, val_fc_before, g_before)

            # log-likelihood for observed action event
            if action_id in self.action_predicates:
                k_idx = self.action_predicates.index(action_id)
                lam = lambda_k[k_idx]
                lam = torch.clamp(lam, min=1e-8)
                total_loglik = total_loglik + torch.log(lam)

            total_survival = total_survival + delta_t * torch.sum(lambda_k)

            # append the action to history (it is now observed at t_i)
            history.append((action_id, t_i))
            # build val after inserting action into history (so engine can deduce consequents)
            val_after_insert = self.build_val_from_history(history, M_minus, t_i, recent_window)

            # run forward chaining after inserting action (this yields immediate logical consequences)
            val_post, g_post, trace = self.forward_chaining_all(val_after_insert, M_minus, max_iters=6, return_trace=True)

            event_trace_list.append({
                'time': t_i,
                'action': action_id,
                'trace': trace
            })

            # update mental M: NOTE we use S_post from g_post; here I keep the same integration window
            # (this mirrors earlier code) but you could integrate with a smaller dt if you prefer.
            S_post = self.compute_S(g_post)
            # integrate from just-before-event to just-after-event using same delta_t (consistent with prior code)
            M_plus = M_minus + delta_t * (-alpha_vec * M_minus + (1.0 - M_minus) * S_post)
            M_plus = torch.clamp(M_plus, 0.0, 1.0)

            # step forward
            M = M_plus
            t_prev = t_i

        if return_trace:
            global_chain = self.extract_single_global_chain_from_events(event_trace_list, g_threshold=g_threshold, delta_threshold=delta_threshold)
            return total_loglik, total_survival, event_trace_list, global_chain
        else:
            return total_loglik, total_survival

    # ------------------------------------------------------------------
    # Single global chain extraction (improved)
    # ------------------------------------------------------------------
    def extract_single_global_chain_from_events(self, event_trace_list, g_threshold=0.05, delta_threshold=1e-4):
        """
        Build single connected chain across events.

        Changes vs earlier:
         - final_head_val uses the max(new_val) seen for that head (more inclusive).
         - Candidate pool includes clauses if:
             * clause g >= tiny_g_inclusion OR
             * clause delta >= delta_threshold OR
             * the head's final_head_val >= g_threshold
           (i.e. we don't drop clauses just because their immediate g is small; we consider head-level aggregated new_val)
         - Greedy selection: choose candidate by (final_head_val[head], clause g) descending.
        """
        global_chain = []
        tiny_g_inclusion = 1e-8  # include very small candidates for trace-based chain construction

        for ev in event_trace_list:
            t = float(ev.get('time', 0.0))
            action = int(ev.get('action'))
            trace = ev.get('trace', []) or []

            # start with the action (string for readability)
            global_chain.append(f"action:{action}@{t:.3f}")
            active_set = set([action])

            # group trace entries by head
            head_map = {}
            for inf in trace:
                try:
                    head = int(inf.get('head'))
                except Exception:
                    # if head stored as non-int, skip
                    continue
                head_map.setdefault(head, []).append(inf)

            # compute final_head_val per head (use max of new_val across records; inclusive)
            final_head_val = {}
            for head, recs in head_map.items():
                new_vals = [float(r.get('new_val', 0.0)) for r in recs]
                if new_vals:
                    final_head_val[head] = max(new_vals)
                else:
                    final_head_val[head] = 0.0

            # build candidate list (per-clause-record)
            cand = []
            for inf in trace:
                g = float(inf.get('g', 0.0))
                prev_val = float(inf.get('prev_val', 0.0))
                new_val = float(inf.get('new_val', 0.0))
                delta = new_val - prev_val
                head = int(inf.get('head'))
                matched_raw = inf.get('matched_predicates', []) or []
                # matched might be predicate ids already; coerce to ints where possible
                matched = []
                for p in matched_raw:
                    try:
                        matched.append(int(p))
                    except Exception:
                        # ignore non-numeric matched entries
                        pass

                # inclusion rule: more inclusive than before
                include = (g >= tiny_g_inclusion) or (delta >= delta_threshold) or (final_head_val.get(head, 0.0) >= g_threshold)
                if include:
                    cand.append({
                        'head': head,
                        'matched': matched,
                        'g': g,
                        'delta': delta,
                        'new_val': new_val,
                        'raw': inf
                    })

            used_heads = set()
            # greedy expansion within this event
            while True:
                # candidates that connect to active_set and not used
                candidates = [c for c in cand if (set(c['matched']) & active_set) and (c['head'] not in used_heads)]
                if not candidates:
                    break
                # choose by final_head_val (primary) then clause g (secondary)
                candidates.sort(key=lambda x: (final_head_val.get(x['head'], 0.0), x['g']), reverse=True)
                chosen = candidates[0]
                head = chosen['head']
                tag = "action" if head in self.action_predicates else ("mental" if head in self.mental_predicates else "pred")
                global_chain.append(f"{tag}:{head}@{t:.3f}")
                # add the newly activated head to active set so next steps can match it
                active_set.add(head)
                used_heads.add(head)

            # event done; go to next (do not carry active_set forward)
        return global_chain

    def extract_global_chains(self, event_trace_list, g_threshold=0.05, delta_threshold=1e-4):
        chain = self.extract_single_global_chain_from_events(event_trace_list, g_threshold=g_threshold, delta_threshold=delta_threshold)
        return [chain] if chain else []
