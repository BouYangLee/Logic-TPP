import numpy as np
import itertools
import matplotlib.pyplot as plt
import os
import pickle
np.random.seed(1024)
import torch
import torch.nn.functional as F


class Model:

    def __init__(self, time_tolerance, decay_rate, sep):

        ### The following parameters are used to manually define the logic rules
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.Time_tolerance = time_tolerance
        self.mental_predicate_set = [1]
        self.action_predicate_set = [2, 3]
        self.head_predicate_set = [1, 2, 3]  ### The index set of all head predicates
        self.total_predicate_set = [1, 2, 3]
        self.decay_rate = decay_rate ### Decay kernel
        self.sep = sep

        ### The following parameters are used to generate synthetic data
        ### for the learning part, the following is used to claim variables
        ### self.model_parameter = {0:{}, 1:{}, ..., 6:{}}
        self.model_parameter = {}

        '''
        Mental
        '''
        head_predicate_idx = 1
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = 0.10

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.60]

        '''
        Action
        '''
        head_predicate_idx = 2
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = 0.30

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.60]


        head_predicate_idx = 3
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = 0.30
        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.60]

        ### NOTE: Set the content of logic rules
        self.logic_template = self.logic_rule()

    def logic_rule(self):
        '''
        This function encodes the content of logic rules
        logic_template = {0:{}, 1:{},..., 6:{}}
        '''

        '''
        Only head predicates may have negative values, because we only record each body predicate's boosted time 
        (states of body predicates are always be 1). Body predicates must happen before head predicate in the same logic rule
        '''

        logic_template = {}

        '''
        Mental predicate: [1]
        '''

        head_predicate_idx = 1
        logic_template[head_predicate_idx] = {}

        ### NOTE: Rule content: 2 and before(2, 1) to 1
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1] ### Use 1 to indicate True; use 0 to indicate False
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1] ### predefine the head predicate is true?
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2, 1]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        '''
        Action predicates: [2, 3]
        '''

        head_predicate_idx = 2
        logic_template[head_predicate_idx] = {}

        ### NOTE: Rule content: 3 and before(3, 2) to 2
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [3]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[3, 2]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        head_predicate_idx = 3
        logic_template[head_predicate_idx] = {}

        ### NOTE: Rule content: 1 and before(1, 3) to 3
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1, 3]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        return logic_template

    def intensity(self, cur_time, head_predicate_idx, history, mental_state):
        truth_formula = []
        weight_intensity = []
        effect_formula = []
        weight_intensity_mental = []
        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            ### Range all the formula for the chosen head_predicate
            cur_weight = self.model_parameter[head_predicate_idx][formula_idx]['weight_para'][0]
            weight_intensity.append(cur_weight)
            truth_formula.append(self.get_truth(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                    history=history,
                                                    template=self.logic_template[head_predicate_idx][formula_idx]))
            effect_formula.append(self.get_formula_effect(template=self.logic_template[head_predicate_idx][formula_idx]))

        intensity_logic = np.array(weight_intensity) * np.array(truth_formula) * np.array(effect_formula)
        intensity_mental = F.softplus(self.model_parameter[head_predicate_idx]['base'] + torch.dot(weight_intensity_mental, mental_state))
        intensity = intensity_mental + np.sum(intensity_logic)
        if intensity >= 0:
            intensity = np.max([intensity, self.model_parameter[head_predicate_idx]['base']])
        else:
            ### TODO: in this case, the intensity with respect to neg effect will always be positive,
            ### and it maybe even bigger than some intensity correspond to positive effect
            intensity = np.exp(intensity)
        return intensity

    ### get_formula_effect() is to compute soft-logic truth value g_f of rule f, return a number in range [0,1] to the truth of head_predicate of rule f
    def get_intensity_truth(self, cur_time, head_predicate_idx, history, template):

        return truth

    '''
    function 'get_truth()': soft-logic computation
    '''

    ### compute the existence of rule
    def get_formula_effect(self, template):
        if template['head_predicate_sign'][0] == 1:
            formula_effect = 1
        else:
            formula_effect = -1  ### why -1?
        return formula_effect

    '''
    function 'get_formula_effect()': compute rule existence?
    '''

    def ODE_mental(self, cur_time, mental_state, history):

        return mental_state

    '''
    function 'ODE_mental()': to compute mental_state at current time
    '''

    def get_ODE_truth(self, cur_time, head_predicate_idx, history, template):

        return truth