# constants.py

# 时序关系常量（与样例保持一致）
BEFORE = "BEFORE"
EQUAL  = "EQUAL"
AFTER  = "AFTER"

# 谓词集合
mental_predicate_set = [1, 2, 3, 4, 9]     # mental
action_predicate_set = [5, 6, 7, 8]        # action
head_predicate_set   = [1, 2, 3, 4, 5, 6, 7, 8, 9]
total_predicate_set  = mental_predicate_set + action_predicate_set

PAD = 0
grid_length = 0.5

# soft-logic 默认温度
DEFAULT_D = 6
DEFAULT_T_MATCH = 0.6
DEFAULT_TAU = 0.1
DEFAULT_BETA = 5.0

# 规则阈值
DEFAULT_G_THRESHOLD = 0.05
DEFAULT_DELTA_THRESHOLD = 1e-4
