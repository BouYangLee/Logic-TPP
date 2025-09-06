# debug_demo.py
import sys
sys.path.append('.')  # adjust if needed
from models.model import LA_TPP_Model, Rule
import constants as C

rules = [
    Rule(head=4, body_predicate_idx=[5,2,3], rule_type='M->M', name='5_2_3->4'),
    Rule(head=4, body_predicate_idx=[5,7],   rule_type='A->M', name='5_7->4'),
    Rule(head=2, body_predicate_idx=[7,4],   rule_type='M->M', name='7_4->2'),
    Rule(head=3, body_predicate_idx=[2,4],   rule_type='M->M', name='2_4->3'),
    Rule(head=7, body_predicate_idx=[2,3], rule_type='M->M', name='2_3->7'),
    Rule(head=1, body_predicate_idx=[3,7,6],   rule_type='A->M', name='3_7_6->1'),
]
# rules = [
#     Rule(head=4, body_predicate_idx=[5,2,3], rule_type='M->M', name='5_2_3->4'),
#     Rule(head=3, body_predicate_idx=[1,6],   rule_type='A->M', name='1_6->3'),
#     Rule(head=2, body_predicate_idx=[7,4],   rule_type='M->M', name='7_4->2'),
#     Rule(head=4, body_predicate_idx=[1,3],   rule_type='M->M', name='1_3->4'),
#     Rule(head=5, body_predicate_idx=[3,4],   rule_type='M->M', name='3_4->5'),
#     Rule(head=3, body_predicate_idx=[7,5,4],   rule_type='A->M', name='7_5_4->3'),
# ]

model = LA_TPP_Model(rules, list(C.mental_predicate_set), list(C.action_predicate_set), device='cpu')

seq = [(5,0.0),(7,0.0),(6,0.5)]
#seq = [(1,0.0),(6,0.0),(7,0.5)]
ll, integ = model.handle_event_sequence(seq)
print("Global chain:", model.last_debug['global_chain'])
