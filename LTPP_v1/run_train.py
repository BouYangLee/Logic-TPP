# run_train.py
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from constants import mental_predicate_set, action_predicate_set, total_predicate_set
from dataset import EventData, collate_fn
from models.model import LA_TPP_Model, Rule
from trainer import Trainer
from utils import plot_loss, plot_W_history
import torch.nn.functional as F

# --- Synthetic sequence generator (keeps same format used elsewhere) ---
def make_seq(n_events=5, t0=0.0):
    seq = []; t = t0
    for i in range(n_events):
        t += random.random() * 0.4 + 0.05
        act = 5 if (i % 2 == 0 and random.random() < 0.75) else 6 if (i % 2 == 1 and random.random() < 0.75) else random.choice([5,6])
        seq.append({'time_since_start': t, 'type_event': act})
        if random.random() < 0.6:
            if random.random() < 0.6: seq.append({'time_since_start': t + 0.02, 'type_event': 1})
            if random.random() < 0.45: seq.append({'time_since_start': t + 0.03, 'type_event': 3})
            if random.random() < 0.3: seq.append({'time_since_start': t + 0.04, 'type_event': 2})
    return sorted(seq, key=lambda x: x['time_since_start'])

def build_dataset(n_seqs=12, seq_len=5):
    return [make_seq(seq_len) for _ in range(n_seqs)]

# --- Helper: convert a full seq (list of dicts) to action-only event list used by handle_event_sequence ---
def seq_dicts_to_action_list(seq_dicts, action_predicate_set):
    """Return list of (action_id, time) for action events in seq_dicts, in chronological order."""
    out = []
    for e in sorted(seq_dicts, key=lambda x: x['time_since_start']):
        if e['type_event'] in action_predicate_set:
            out.append((int(e['type_event']), float(e['time_since_start'])))
    return out

# --- Helper: sanitize trace for JSON dump (remove heavy W if present, convert numbers) ---
def sanitize_event_traces_for_json(event_traces):
    sanitized = []
    for ev in event_traces:
        ev_s = {
            'time': float(ev.get('time', 0.0)),
            'action': int(ev.get('action', -1)),
            'chains': ev.get('chains', []),
        }
        # keep only compact trace summary (without heavy W)
        trace_list = []
        for inf in ev.get('trace', []):
            inf_s = {
                'iter': int(inf.get('iter', 0)),
                'clause_key': inf.get('clause_key'),
                'head': inf.get('head'),
                'g': float(inf.get('g', 0.0)),
                'matched_predicates': inf.get('matched_predicates', []),
                'prev_val': float(inf.get('prev_val', 0.0)),
                'new_val': float(inf.get('new_val', 0.0))
            }
            trace_list.append(inf_s)
        ev_s['trace'] = trace_list
        sanitized.append(ev_s)
    return sanitized

def main():
    outdir = 'outputs'
    os.makedirs(outdir, exist_ok=True)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # build small synthetic dataset
    data = build_dataset(n_seqs=12, seq_len=5)
    ds = EventData(data)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, shuffle=True, num_workers=0)

    # define rules (example complex rules)
    rules = [
        Rule(head=4, body_predicate_idx=[5, 2, 3], rule_type='M->M', name='5_2_3->4'),  # 5 and 2 and 3 ->4
        Rule(head=4, body_predicate_idx=[5, 1], rule_type='M->M', name='5_1->4'),  # 5 and 1 ->4
        Rule(head=2, body_predicate_idx=[1, 4], rule_type='M->M', name='1_4->2'),  # 1 and 4 ->2
        Rule(head=6, body_predicate_idx=[2, 1, 4], rule_type='M->A', name='2_1_4->6'),  # 2 and 1 and 4 ->6
        Rule(head=1, body_predicate_idx=[3, 6], rule_type='A->M', name='3_6->1')  # 3 and 6 ->1
    ]

    device = 'cpu'
    model = LA_TPP_Model(rules=rules, mental_predicates=mental_predicate_set,
                         action_predicates=action_predicate_set,
                         predicate_list=total_predicate_set, d_pred=4, device=device,
                         learnable_K=False)

    # print("=== PARAMETER CHECK ===")
    # for name, p in model.named_parameters():
    #     print(name, "requires_grad=", p.requires_grad, "shape=", tuple(p.shape))
    # print("========================")

    trainer = Trainer(model, lr=5e-3, device=device)



    n_epochs = 50
    eval_interval = 5   # every N epochs run trace-enabled evaluation on 1 example sequence
    example_idx = 0     # which sequence from the synthetic data to use for trace debugging

    losses = []
    # use param string names for W_history keys
    param_names = [model.key_to_str[k] for k in model.clause_keys]
    W_history = {name: [] for name in param_names}

    print("Starting training ...")
    for ep in range(n_epochs):
        avg_loss = trainer.train_epoch(loader)
        losses.append(avg_loss)

        # record W_history for diagnostics
        for key in model.clause_keys:
            param_name = model.key_to_str[key]
            Theta = model.get_theta(key).detach()
            Kn = F.normalize(model.K.detach(), dim=1)
            Thetan = F.normalize(Theta, dim=0)
            S = Kn @ Thetan
            W = F.softmax(S / model.engine.T_match, dim=0)
            W_sum = W.sum(dim=1)
            W_history[param_name].append(W_sum.cpu().numpy())

        print(f"Epoch {ep+1}/{n_epochs}, avg loss={avg_loss:.4f}")

        # periodic evaluation: export trace/chains on a small example (no grads)
        if (ep + 1) % eval_interval == 0 or ep == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                example_seq_dicts = data[example_idx]
                example_action_list = seq_dicts_to_action_list(example_seq_dicts, action_predicate_set)
                # call with return_trace=True to get chain-level info
                try:
                    # loglik, surv, event_traces = model.handle_event_sequence(example_action_list, return_trace=True)
                    res = model.handle_event_sequence(example_action_list, return_trace=True, g_threshold=0.02, delta_threshold=1e-6)
                    # res 可能是:
                    #  - (loglik, surv)                          -> len == 2
                    #  - (loglik, surv, event_traces)            -> len == 3
                    #  - (loglik, surv, event_traces, global_chain) -> len == 4
                    if isinstance(res, tuple):
                        if len(res) == 2:
                            loglik, surv = res
                            event_traces = []
                            global_chain = None
                        elif len(res) == 3:
                            loglik, surv, event_traces = res
                            global_chain = None
                        elif len(res) == 4:
                            loglik, surv, event_traces, global_chain = res
                        else:
                            raise ValueError(
                                f"Unexpected return tuple shape from handle_event_sequence: len={len(res)}")
                    else:
                        # 如果模型直接返回非 tuple（不太可能），把它包装/报错
                        raise ValueError("Unexpected non-tuple return from handle_event_sequence()")
                except TypeError:
                    # in case model.handle_event_sequence doesn't accept return_trace (backwards compat),
                    # fallback to calling and ignoring trace (should not happen if model updated)
                    res = model.handle_event_sequence(example_action_list)
                    if len(res) == 3:
                        loglik, surv, event_traces = res
                    else:
                        loglik, surv = res
                        event_traces = []

                # event_traces: list of dicts {'time', 'action', 'trace', 'chains'}
                # sanitize & print summary
                print("=== Trace eval (epoch {}) ===".format(ep+1))
                for ev in event_traces:
                    t = ev.get('time', None)
                    a = ev.get('action', None)
                    print(f"Event time={t:.3f}, action={a}, chains found={len(ev.get('chains', []))}")
                    for ch in ev.get('chains', []):
                        print("   chain: " + " -> ".join(ch))
                # save sanitized JSON for later inspection
                sanitized = sanitize_event_traces_for_json(event_traces)
                outpath = os.path.join(outdir, f"trace_epoch_{ep+1}.json")
                with open(outpath, 'w') as fh:
                    json.dump({'epoch': ep+1, 'loglik': float(loglik), 'survival': float(surv), 'event_traces': sanitized}, fh, indent=2)
                print(f"Trace saved to {outpath}")
            model.train()

    # final plots
    plot_loss(losses, outpath=os.path.join(outdir, 'loss.png'))
    plot_W_history(W_history, predicate_list=total_predicate_set, outdir=outdir)
    print("Done. Plots and traces saved to", outdir)

    # # test data
    # with torch.no_grad():
    #     # 固定例子： t=0 有 action 5 和 observed mental 1； t=0.5 有 observed mental 3
    #     example_seq = [
    #         {'time_since_start': 0.0, 'type_event': 5},  # action 5 @ t=0
    #         {'time_since_start': 0.0, 'type_event': 1},  # observed mental 1 @ t=0
    #         {'time_since_start': 0.5, 'type_event': 3}  # observed mental 3 @ t=0.5
    #     ]
    #     # 把 action-only 列表做出来（handle_event_sequence 期望的格式）
    #     example_action_list = seq_dicts_to_action_list(example_seq, action_predicate_set)
    #     # observed mental events 列表 (pred, time)
    #     observed_mentals = [(e['type_event'], e['time_since_start']) for e in example_seq if
    #                         e['type_event'] in mental_predicate_set]
    #
    #     # 调用模型（要求 model.handle_event_sequence 接受 observed_mentals 参数并在 return_trace=True 时返回 (loglik, surv, event_traces, global_chain)）
    #     try:
    #         res = model.handle_event_sequence(example_action_list, return_trace=True, observed_mentals=observed_mentals, g_threshold=0.02, delta_threshold=1e-6)
    #     except TypeError:
    #         # 若你的 model.handle_event_sequence 尚未接受 observed_mentals 参数，
    #         # 尝试不传 observed_mentals（或把 observed mentals 自行插入 history —— 见下方说明）
    #         res = model.handle_event_sequence(example_action_list, return_trace=True, g_threshold=0.02, delta_threshold=1e-6)
    #
    #     # res 可能是:
    #     #  - (loglik, surv)                          -> len == 2
    #     #  - (loglik, surv, event_traces)            -> len == 3
    #     #  - (loglik, surv, event_traces, global_chain) -> len == 4
    #     if isinstance(res, tuple):
    #         if len(res) == 2:
    #             loglik, surv = res
    #             event_traces = []
    #             global_chain = None
    #         elif len(res) == 3:
    #             loglik, surv, event_traces = res
    #             global_chain = None
    #         elif len(res) == 4:
    #             loglik, surv, event_traces, global_chain = res
    #         else:
    #             raise ValueError(f"Unexpected return tuple shape from handle_event_sequence: len={len(res)}")
    #     else:
    #         # 如果模型直接返回非 tuple（不太可能），把它包装/报错
    #         raise ValueError("Unexpected non-tuple return from handle_event_sequence()")
    #
    #     # 打印可读信息
    #     print("=== Single-run Trace Debugging ===")
    #     for ev in event_traces:
    #         t = ev.get('time', None)
    #         a = ev.get('action', None)
    #         print(
    #             f"Event time={t:.3f}, action={a}, trace entries={len(ev.get('trace', []))}, chains found={len(ev.get('chains', [])) if 'chains' in ev else 'N/A'}")
    #         if 'chains' in ev:
    #             for ch in ev['chains']:
    #                 print("   chain: " + " -> ".join(ch))
    #     if global_chain is not None:
    #         print("=== Global chain (single list) ===")
    #         print(" -> ".join(global_chain))
    # debug-forward: do NOT disable grad so we can inspect grad_fn on tensors
    # with torch.enable_grad():
    #     example_seq = [
    #         {'time_since_start': 0.0, 'type_event': 5},  # action 5 @ t=0
    #         {'time_since_start': 0.0, 'type_event': 1},  # observed mental 1 @ t=0
    #         {'time_since_start': 0.5, 'type_event': 3}  # observed mental 3 @ t=0.5
    #     ]  # 你已有的例子
    #     example_action_list = seq_dicts_to_action_list(example_seq, action_predicate_set)
    #     val_pre = model.build_val_from_history([], torch.zeros(len(model.mental_predicates)), 0.0)
    #     # call forward_chaining_all to get val,g_cache,trace but with grad enabled
    #     val_out, g_cache, trace = model.forward_chaining_all(val_pre, torch.zeros(len(model.mental_predicates)),
    #                                                          return_trace=True)
    #     print("=== DEBUG forward_chaining_all g_cache sample ===")
    #     for k, g in list(g_cache.items())[:6]:
    #         print("key:", k, "type:", type(g), "is tensor?", isinstance(g, torch.Tensor))
    #         if isinstance(g, torch.Tensor):
    #             print("  g shape:", tuple(g.shape))
    #             print("  g requires_grad:", g.requires_grad)
    #             print("  g.grad_fn:", g.grad_fn)
    #     # check Theta requires_grad for first few clauses
    #     print("=== Theta requires_grad sample ===")
    #     for k in list(model.Thetas.keys())[:6]:
    #         print(k, "requires_grad=", model.Thetas[k].requires_grad)
    # ---------- debug helper: inspect single-run traces ----------
    with torch.no_grad():
        example_seq = [
            {'time_since_start': 0.0, 'type_event': 5},  # action 5 @ t=0
            {'time_since_start': 0.0, 'type_event': 1},  # observed mental 1 @ t=0
            {'time_since_start': 0.5, 'type_event': 3}  # observed mental 3 @ t=0.5
        ]
        example_action_list = seq_dicts_to_action_list(example_seq, action_predicate_set)
        observed_mentals = [(e['type_event'], e['time_since_start']) for e in example_seq if
                            e['type_event'] in mental_predicate_set]

        # 调用模型，拿 trace（不要只看 global_chain）
        res = model.handle_event_sequence(example_action_list, return_trace=True)
        # 如果你的 handle_event_sequence 返回4项 (loglik,surv,event_traces,global_chain)
        if isinstance(res, tuple) and len(res) >= 3:
            loglik, surv, event_traces = res[0], res[1], res[2]
        else:
            print("Unexpected return from handle_event_sequence:", type(res), res)
            event_traces = []

        print("=== SINGLE-RUN DEBUG ===")
        for ev_idx, ev in enumerate(event_traces):
            t = ev.get('time', None)
            a = ev.get('action', None)
            trace = ev.get('trace', [])
            print(f"\nEvent {ev_idx}: time={t:.3f} action={a} trace_entries={len(trace)}")
            # 打印前若干 trace 条目（看 g, candidate_val, new_val, matched）
            for i, inf in enumerate(trace[:30]):
                # 安全打印单条 inference 的小工具
                def fmt_num(x, fmt="{:.6f}"):
                    if x is None:
                        return "None"
                    try:
                        return fmt.format(float(x))
                    except Exception:
                        return str(x)

                # 假设你在单次 debug 部分有类似下面的循环，替换为：
                for i, inf in enumerate(ev.get('trace', [])):
                    head = inf.get('head', None)
                    clause = inf.get('clause_key', None)
                    g_val = inf.get('g', None)
                    prev = inf.get('prev_val', None)
                    cand = inf.get('candidate', None) if 'candidate' in inf else inf.get('new_val', None)  # 兼容字段
                    newv = inf.get('new_val', None)

                    # 如果需要把 matched_predicates 打出来：
                    matched = inf.get('matched_predicates', None)

                    print(f"  [{i}] head={head}, clause={clause}, g={fmt_num(g_val)}, "
                          f"prev={fmt_num(prev)}, candidate={fmt_num(cand)}, new={fmt_num(newv)}, matched={matched}")

            # 下面复刻 extract_single_global_chain_from_events 的头部计算，打印中间值
            # group by head:
            head_map = {}
            for inf in trace:
                head_map.setdefault(int(inf['head']), []).append(inf)
            print("  final_head_val summary:")
            for head, recs in head_map.items():
                updates = [r for r in recs if (float(r.get('new_val', 0.0)) - float(r.get('prev_val', 0.0))) > 1e-8]
                if updates:
                    fhv = max(float(r.get('new_val', 0.0)) for r in updates)
                    note = "has_updates"
                else:
                    fhv = max(float(r.get('g', 0.0)) for r in recs)
                    note = "no_updates, fallback to max_g"
                print(f"    head={head}: final_head_val={fhv:.6f} ({note}), n_recs={len(recs)}")
            # build candidate list (as extract does)
            cand = []
            for inf in trace:
                g = float(inf.get('g', 0.0))
                delta = float(inf.get('new_val', 0.0)) - float(inf.get('prev_val', 0.0))
                if (g >= 0.01) or (delta >= 1e-6):  # 临时放低阈值便于 debug
                    cand.append(
                        {'head': int(inf['head']), 'g': g, 'delta': delta, 'matched': inf.get('matched_predicates')})
            print("  candidates (g>=0.01 or delta>=1e-6): count=", len(cand))
            # 打印 top 10 candidates
            for i, c in enumerate(sorted(cand, key=lambda x: (x['g'], x['delta']), reverse=True)[:10]):
                print(f"    top[{i}] head={c['head']}, g={c['g']:.6f}, delta={c['delta']:.6f}, matched={c['matched']}")
        # print global_chain if exists
        if isinstance(res, tuple) and len(res) == 4:
            print("\nGlobal chain returned by model:")
            print(" -> ".join(res[3]))
        print("\n=== END DEBUG ===\n")


if __name__ == '__main__':
    main()

