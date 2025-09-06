import math
import torch
import torch.nn.functional as F
from torch import nn

class SoftLogicEngine(nn.Module):
    def __init__(self, predicate_list, d=None, K_init=None, T_match=0.8, tau=0.05, beta=8.0):
        super().__init__()
        self.predicate_list = list(predicate_list)
        self.P = len(self.predicate_list)
        self.pred_to_pos = {p: i for i, p in enumerate(self.predicate_list)}
        self.T_match = T_match
        self.tau = tau
        self.beta = beta
        if K_init is not None:
            self.d = K_init.shape[1]
            self.K = nn.Parameter(K_init.clone(), requires_grad=False)
        else:
            self.d = self.P if d is None else d
            self.K = nn.Parameter(torch.randn(self.P, self.d) * 0.1, requires_grad=False)

    @staticmethod
    def softmin_tau(x, tau):
        """
        differentiable soft approximation of min:
        softmin_tau(x) = -(1/tau) * ( logsumexp(-x/tau) - log(n) )
        """
        if x.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        n = x.shape[-1]
        return -(1/tau) * (torch.logsumexp(-x / tau, dim=-1) - math.log(n))

    # --- soft-matching 版本：我们在训练可学习匹配时用；本项目主路径用 exact 版本 ---
    def eval_single_clause(self, Theta_f, val_vec, T_match=None, tau=None):  # soft-and
        if T_match is None:
            T_match = self.T_match
        if tau is None:
            tau = self.tau
        Kn = F.normalize(self.K, dim=1)  # (P, d)
        Thetan = F.normalize(Theta_f, dim=0)  # (d, h)
        S = Kn @ Thetan  # (P, h)
        W = F.softmax(S / (T_match + 1e-9), dim=0)  # (P, h)
        conf = (W * S).sum(dim=0)  # (h,)
        matched = (W * val_vec.unsqueeze(1)).sum(dim=0)  # (h,)
        x = torch.cat([conf, matched], dim=0)
        g_and = self.softmin_tau(x, tau)
        return torch.clamp(g_and, 0.0, 1.0)

    def soft_or(self, g_list, beta=None):
        if beta is None:
            beta = self.beta
        if len(g_list) == 0:
            return torch.tensor(0.0)
        g_stack = torch.stack(g_list)
        return (1.0 / beta) * torch.logsumexp(beta * g_stack, dim=0)

    # --- exact 版本：body id 精确匹配 ---
    def eval_clause_exact(self, body_pred_ids, val_vec, tau=None, min_body_eps=1e-8):
        if tau is None:
            tau = self.tau
        if len(body_pred_ids) == 0:
            return torch.tensor(0.0, device=val_vec.device)

        vals = []
        for pid in body_pred_ids:
            pos = self.pred_to_pos.get(pid, None)
            if pos is None:
                # 未知谓词：视为 0，下面会被短路
                vals.append(torch.tensor(0.0, device=val_vec.device))
            else:
                vals.append(val_vec[pos])
        x = torch.stack(vals)

        # 任一体谓词为 0，则直接短路为 0（防止噪声触发）
        if torch.min(x).item() <= min_body_eps:
            return torch.tensor(0.0, device=val_vec.device)

        g_and = self.softmin_tau(x, tau)
        return torch.clamp(g_and, 0.0, 1.0)

    # --- 兼容接口：带 trace 的 soft-matching 版本（当前不在主路径里用） ---
    def forward_chaining_with_trace(self, Thetas, initial_val, max_iters=6, tol=1e-6,
                                    return_W=False, debug_verbose=False):
        device = initial_val.device
        val = initial_val.clone().to(device)
        g_cache = {}
        head_to_clauses = {}
        for k in Thetas.keys():
            head = k[0]
            head_to_clauses.setdefault(head, []).append(k)

        trace_list = []
        for it in range(max_iters):
            changed = False
            val_new = val.clone()

            for head, clause_keys in head_to_clauses.items():
                clause_infos = []
                for key in clause_keys:
                    Theta_f = Thetas[key].to(device)
                    g, W, argmax_preds = self.eval_single_clause_with_match(Theta_f, val, debug_verbose=debug_verbose)
                    clause_infos.append((key, g, W, argmax_preds))
                g_list = [ci[1] for ci in clause_infos]
                g_stack = torch.stack(g_list)
                if g_stack.shape[0] == 1:
                    g_head = g_stack[0]
                else:
                    beta_effective = max(self.beta, 1.0)
                    g_head = (1.0 / beta_effective) * torch.logsumexp(beta_effective * g_stack, dim=0)

                pos = self.pred_to_pos[head]
                old_val = float(val[pos].item())
                new_val = max(old_val, float(g_head.item()))
                delta = new_val - old_val
                if delta > tol:
                    val_new[pos] = new_val
                    changed = True

                for key, g, W, argmax in clause_infos:
                    g_cache[key] = float(g.item())

            val = val_new
            if not changed:
                break

        return val, g_cache, trace_list

    def eval_single_clause_with_match(self, Theta_f, val, debug_verbose=False):
        Kn = F.normalize(self.K, dim=1)  # (P, d)
        Thetan = F.normalize(Theta_f, dim=0)  # (d, h)
        S = Kn @ Thetan  # (P, h)

        h = S.shape[1]
        argmax_predicates = []
        slot_confidences = []
        slot_values = []

        for slot_idx in range(h):
            sim_scores = S[:, slot_idx]
            best_pred_pos = torch.argmax(sim_scores)
            best_pred_id = self.predicate_list[int(best_pred_pos)]
            argmax_predicates.append(best_pred_id)
            confidence = sim_scores[best_pred_pos]
            truth_value = val[best_pred_pos]
            slot_confidences.append(confidence)
            slot_values.append(truth_value)

        conf_tensor = torch.stack(slot_confidences)
        val_tensor = torch.stack(slot_values)
        combined = torch.cat([conf_tensor, val_tensor], dim=0)
        g_and = self.softmin_tau(combined, self.tau)
        return torch.clamp(g_and, 0.0, 1.0), None, argmax_predicates
