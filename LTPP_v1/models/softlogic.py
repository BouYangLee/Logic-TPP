# models/softlogic.py
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
        # differentiable soft approximation of min
        n = x.shape[-1]
        return -tau * (torch.logsumexp(-x / tau, dim=-1) - math.log(n))

    def eval_single_clause(self, Theta_f, val_vec, T_match=None, tau=None):
        """
        Original simple evaluator returning g_f^{AND} scalar.
        """
        if T_match is None:
            T_match = self.T_match
        if tau is None:
            tau = self.tau
        Kn = F.normalize(self.K, dim=1)       # (P, d)
        Thetan = F.normalize(Theta_f, dim=0)  # (d, h)
        S = Kn @ Thetan                        # (P, h)
        W = F.softmax(S / T_match, dim=0)      # (P, h)
        conf = (W * S).sum(dim=0)              # (h,)
        matched = (W * val_vec.unsqueeze(1)).sum(dim=0)  # (h,)
        x = torch.cat([conf, matched], dim=0)
        g_and = self.softmin_tau(x, tau)
        return torch.clamp(g_and, 0.0, 1.0)

    def soft_or(self, g_list, beta=None):
        """
        Soft OR (LogSumExp) over a list of scalar tensors.
        """
        if beta is None:
            beta = self.beta
        g_stack = torch.stack(g_list)
        return (1.0 / beta) * torch.logsumexp(beta * g_stack, dim=0)

    # -----------------------------
    # Matching + trace-enabled evals
    # -----------------------------
    def eval_single_clause_with_match(self, Theta_f, val_vec, T_match=None, tau=None):
        """
        Like eval_single_clause but also returns per-slot soft-assignment W and argmax matches.
        Returns: (g_and (scalar tensor), W (tensor P×h), argmax_predicates (list of predicate ids))
        """
        if T_match is None:
            T_match = self.T_match
        if tau is None:
            tau = self.tau

        Kn = F.normalize(self.K, dim=1)       # (P, d)
        Thetan = F.normalize(Theta_f, dim=0)  # (d, h)
        S = Kn @ Thetan                        # (P, h)
        W = F.softmax(S / T_match, dim=0)      # (P, h)

        conf = (W * S).sum(dim=0)              # (h,)
        matched = (W * val_vec.unsqueeze(1)).sum(dim=0)  # (h,)
        x = torch.cat([conf, matched], dim=0)
        g_and = self.softmin_tau(x, tau)

        # ----------------------------
        # IMPORTANT CHANGE:
        # when choosing argmax per slot, bias selection toward predicates that currently
        # have higher truth value (val_vec). This lets a just-fired action (val=1)
        # actually occupy a slot and therefore seed the chain.
        # We still keep W for other computations.
        # ----------------------------
        # small epsilon so that argmax still prefers similarity when val_vec is zero everywhere
        eps = 1e-6
        # val_vec.unsqueeze(1): (P,1). Broadcast with W (P,h)
        bias_matrix = (val_vec.unsqueeze(1) + eps)  # (P,1)
        # combine W with bias to compute argmax scores:
        scored = W * bias_matrix  # (P,h) ; boosts rows where val_vec > 0
        argmax_pos = torch.argmax(scored, dim=0)   # (h,) indices into 0..P-1
        argmax_predicates = [self.predicate_list[int(idx)] for idx in argmax_pos]
        return torch.clamp(g_and, 0.0, 1.0), W, argmax_predicates

    def forward_chaining_with_trace(self, Thetas, initial_val, max_iters=6, tol=1e-6, return_W=False):
        """
        Recursive forward chaining with trace extraction.
        Thetas: dict mapping clause_key (same keys as model.clause_keys) -> Theta_f tensor (d,h)
        initial_val: tensor (P,)
        return_W: if True, records W matrices in trace (heavy). Default False.
        Returns:
            val (P,), g_cache (dict key->float), trace (list of inference events)
        Each trace event is a dict with:
            'iter', 'clause_key', 'head', 'g', 'matched_predicates', 'prev_val', 'new_val'
            optionally 'W' if return_W=True
        """
        device = initial_val.device
        val = initial_val.clone().to(device)
        g_cache = {}
        trace = []
        head_to_clauses = {}
        # group clauses by head predicate
        for k in Thetas.keys():
            head = k[0]
            head_to_clauses.setdefault(head, []).append(k)

        for it in range(max_iters):
            changed = False
            val_new = val.clone()
            for head, clause_keys in head_to_clauses.items():
                clause_infos = []
                # evaluate all clauses for this head
                for key in clause_keys:
                    Theta_f = Thetas[key].to(device)
                    g, W, argmax_preds = self.eval_single_clause_with_match(Theta_f, val)
                    clause_infos.append((key, g, W, argmax_preds))

                # stack g to perform argmax/soft-or safely on device
                g_list = [ci[1] for ci in clause_infos]
                g_stack = torch.stack(g_list)  # (R,)
                # soft-or aggregation (LogSumExp / beta)
                if g_stack.shape[0] == 1:
                    g_head = g_stack[0]
                else:
                    # use engine beta
                    g_head = (1.0 / (self.beta if self.beta is not None else 1.0)) * torch.logsumexp(self.beta * g_stack, dim=0)

                pos = self.pred_to_pos[head]
                old_val = float(val[pos].item())
                new_val = max(old_val, float(g_head.item()))
                if new_val - old_val > tol:
                    val_new[pos] = new_val
                    changed = True
                    max_idx = int(torch.argmax(g_stack).item())
                    key_chosen, g_chosen, W_chosen, argmax_chosen = clause_infos[max_idx]

                    ev = {
                        'iter': int(it),
                        'clause_key': key_chosen,
                        'head': head,
                        'g': float(g_chosen.item()),
                        'matched_predicates': list(argmax_chosen),
                        'prev_val': old_val,
                        'new_val': new_val
                    }
                    if return_W:
                        ev['W'] = W_chosen.detach().cpu().numpy()
                    trace.append(ev)

                # update g_cache for all clauses (store scalar floats)
                for key, g, _, _ in clause_infos:
                    g_cache[key] = float(g.item())

            val = val_new
            if not changed:
                break
        return val, g_cache, trace

    # Backwards-compatible wrapper: returns (val, g_cache) and ignores trace
    def forward_chaining(self, Thetas, initial_val, max_iters=6, tol=1e-6):
        val, g_cache, _ = self.forward_chaining_with_trace(Thetas, initial_val, max_iters=max_iters, tol=tol, return_W=False)
        return val, g_cache
