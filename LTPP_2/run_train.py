import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from constants import mental_predicate_set, action_predicate_set, total_predicate_set
from dataset import EventData, collate_fn
from models.model import LA_TPP_Model, Rule, BEFORE, AFTER, EQUAL
from trainer import Trainer
from utils import plot_loss, plot_W_history
import torch.nn.functional as F

def make_discrete_seq():
    """外生动作：t=0 有 5、7；t=0.5 有 6。"""
    return [
        {'time_since_start': 0.0, 'type_event': 5},
        {'time_since_start': 0.0, 'type_event': 7},
        {'time_since_start': 0.5, 'type_event': 6},
    ]

def build_dataset(n_seqs=10):
    return [make_discrete_seq() for _ in range(n_seqs)]

def seq_dicts_to_action_list(seq_dicts, action_predicate_set):
    out = []
    for e in sorted(seq_dicts, key=lambda x: x['time_since_start']):
        if e['type_event'] in action_predicate_set:
            out.append((int(e['type_event']), float(e['time_since_start'])))
    return out

def sanitize_event_traces_for_json(event_traces):
    sanitized = []
    for ev in event_traces:
        ev_s = {
            'time': float(ev.get('time', 0.0)),
            'actions': list(ev.get('actions', [])),
            'scheduled_merged': list(ev.get('scheduled_merged', [])),
            'inferred_actions_for_next_time': list(ev.get('inferred_actions_for_next_time', [])),
            'mental_states_after': [float(x) for x in ev.get('mental_states_after', [])] if ev.get('mental_states_after') is not None else [],
        }
        trace_list = []
        for inf in ev.get('trace', []):
            inf_s = {
                'iter': int(inf.get('iter', 0)),
                'clause_key': tuple(inf.get('clause_key')) if isinstance(inf.get('clause_key'), (list, tuple)) else inf.get('clause_key'),
                'head': inf.get('head'),
                'g': float(inf.get('g', 0.0)),
                'matched_predicates': inf.get('matched_predicates', []),
                'prev_val': float(inf.get('prev_val', 0.0)),
                'new_val': float(inf.get('new_val', 0.0)),
                'delta': float(inf.get('delta', 0.0))
            }
            trace_list.append(inf_s)
        ev_s['trace'] = trace_list
        sanitized.append(ev_s)
    return sanitized

def analyze_reasoning_logic(event_traces, g_threshold=0.01, delta_threshold=1e-6):
    """
    打印每个时间步：动作（外生 + 合并调度） + 被触发的 mental 链。
    """
    reasoning_chains = []
    if not event_traces:
        return reasoning_chains

    print("=== Reasoning Logic Analysis ===")
    for timestep in event_traces:
        t = float(timestep.get('time', 0.0))
        actions = list(timestep.get('actions', []))
        scheduled_merged = list(timestep.get('scheduled_merged', []))
        trace = timestep.get('trace', []) or []

        merged_actions_for_print = actions + scheduled_merged
        print(f"\nTime t={t:.3f}:")
        print(f"Actions: {merged_actions_for_print}")

        trigger_set = set(merged_actions_for_print)
        activations = []
        seen_heads = set()
        trace_sorted = sorted(trace, key=lambda x: int(x.get('iter', 0)))
        for inf in trace_sorted:
            head = inf.get('head', None)
            if head is None:
                continue
            g = float(inf.get('g', 0.0))
            delta = float(inf.get('delta', 0.0))
            if (g < g_threshold and delta < delta_threshold):
                continue
            if head in seen_heads:
                continue
            matched = set(inf.get('matched_predicates', []) or [])
            if matched & trigger_set:
                seen_heads.add(head)
                activations.append((head, g, delta, int(inf.get('iter', 0)), list(matched)))
                trigger_set.add(head)

        print("Activation chain at this time:")
        for head, g, delta, iter_num, matched in activations:
            print(f"  mental:{head}@{t:.3f}  (g={g:.6f}, Δ={delta:.6f}, iter={iter_num}, matched={matched})")

        chain = []
        for a in merged_actions_for_print:
            chain.append(f"action:{a}@{t:.3f}")
        for head, g, delta, iter_num, matched in activations:
            chain.append(f"mental:{head}@{t:.3f}")

        reasoning_chains.append({
            'time': t,
            'actions': merged_actions_for_print,
            'chain': chain,
            'activations': [{'head': int(h), 'g': float(gv), 'delta': float(dv), 'iter': int(it), 'matched': matched}
                            for (h, gv, dv, it, matched) in activations]
        })
    return reasoning_chains

def main():
    outdir = 'outputs'
    os.makedirs(outdir, exist_ok=True)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # === 规则（结构化时序：不要把 "AFTER(2,4)" 字符串塞进 body 里） ===
    rules = [
        # 5,7 -> 4  （可以附加 EQUAL(5,7) 要求同刻；本例我们不加也能跑）
        Rule(head=4, body_predicate_idx=[5, 7], rule_type='A->M', name='5_7->4',
             temporal_pairs=[(5, 7)], temporal_types=[EQUAL]),

        # 7,4 -> 2  （要求 BEFORE(7,4)）
        Rule(head=2, body_predicate_idx=[7, 4], rule_type='M->M', name='7_4->2',
             temporal_pairs=[(7, 4)], temporal_types=[BEFORE]),

        # 2,4 -> 8  （要求 AFTER(2,4)）
        Rule(head=8, body_predicate_idx=[2, 4], rule_type='M->A', name='2_4->8',
             temporal_pairs=[(2, 4)], temporal_types=[AFTER]),

        # 8,6 -> 2  （A->M，没有时序要求也可以）
        Rule(head=1, body_predicate_idx=[8, 6], rule_type='A->M', name='8_6->1'),
    ]

    print("=== Defined Rules ===")
    for rule in rules:
        print(f"{rule.name}: {rule.body} -> {rule.head} ({rule.type}), temporals={list(zip(rule.temporal_pairs, rule.temporal_types))}")
    print()

    device = 'cpu'
    model = LA_TPP_Model(
        rules=rules,
        mental_predicates=mental_predicate_set,
        action_predicates=action_predicate_set,
        predicate_list=total_predicate_set,
        d_pred=6,
        device=device,
        learnable_K=False
    )

    trainer = Trainer(model, lr=3e-3, device=device)

    n_epochs = 8
    eval_interval = 4

    print("=== Starting Training ===")
    data = build_dataset(n_seqs=6)
    ds = EventData(data)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn, shuffle=True, num_workers=0)

    losses = []
    param_names = [model.key_to_str[k] for k in model.clause_keys]
    W_history = {name: [] for name in param_names}

    for ep in range(n_epochs):
        avg_loss = trainer.train_epoch(loader)
        losses.append(avg_loss)

        # 记录 W 演化（可选）
        for key in model.clause_keys:
            param_name = model.key_to_str[key]
            Theta = model.get_theta(key).detach()
            Kn = F.normalize(model.K.detach(), dim=1)
            Thetan = F.normalize(Theta, dim=0)
            S = Kn @ Thetan
            W = F.softmax(S / 0.6, dim=0)
            W_sum = W.sum(dim=1)
            W_history[param_name].append(W_sum.cpu().numpy())

        print(f"Epoch {ep + 1}/{n_epochs}, avg loss={avg_loss:.4f}")

        if (ep + 1) % eval_interval == 0 or ep == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                example_seq = data[0]
                example_action_list = seq_dicts_to_action_list(example_seq, action_predicate_set)

                print(f"\n=== Evaluation at Epoch {ep + 1} ===")
                print("Example sequence actions:", example_action_list)

                loglik, surv, event_traces, global_chain = model.handle_event_sequence(
                    example_action_list,
                    return_trace=True,
                    g_threshold=0.02,
                    delta_threshold=1e-6,
                    debug_verbose=(ep == n_epochs - 1)
                )

                print(f"Evaluation loglik: {float(loglik):.6f}, survival: {float(surv):.6f}")

                reasoning_chains = analyze_reasoning_logic(event_traces, g_threshold=0.01, delta_threshold=1e-6)

                print("\n=== Reasoning Chains ===")
                for chain_info in reasoning_chains:
                    t = chain_info['time']
                    chain = chain_info['chain']
                    print(f"t={t:.3f}: {' -> '.join(chain)}")

                print("\nGlobal chain:", global_chain)

                if ep == n_epochs - 1:
                    sanitized = sanitize_event_traces_for_json(event_traces)
                    outpath = os.path.join(outdir, f"final_trace_epoch_{ep + 1}.json")
                    with open(outpath, 'w') as fh:
                        json.dump({
                            'epoch': ep + 1,
                            'loglik': float(loglik),
                            'survival': float(surv),
                            'reasoning_chains': reasoning_chains,
                            'event_traces': sanitized
                        }, fh, indent=2)
                    print(f"Detailed trace saved to {outpath}")

            model.train()

    plot_loss(losses, outpath=os.path.join(outdir, 'loss.png'))
    plot_W_history(W_history, predicate_list=total_predicate_set, outdir=outdir)
    print(f"\nTraining completed. Results saved to {outdir}")

if __name__ == '__main__':
    main()
