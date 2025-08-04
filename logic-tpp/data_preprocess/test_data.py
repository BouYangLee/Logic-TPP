# test_data.py
# 这个脚本用于端到端测试 3D 版本的 EventData 和 collate_fn，使用单进程加载（num_workers=0）避免多进程 spawn 问题。

import torch
from torch.utils.data import DataLoader

# 导入你的数据处理模块
from Dataset import EventData, collate_fn
#from model import mental_predicate_set, action_predicate_set, head_predicate_set, total_predicate_set, PAD, time_horizon, grid_length


def main():
    # 1. 构造测试数据，格式同 __init__ 描述
    data = [
        [  # 样本 1
            {'time_since_start': 1.0, 'type_event': 1},
            {'time_since_start': 1.0, 'type_event': 9},
            {'time_since_start': 4.0, 'type_event': 2},
            {'time_since_start': 1.0, 'type_event': 5},  # action
            {'time_since_start': 4.0, 'type_event': 6},  # action
        ],
        [  # 样本 2
            {'time_since_start': 2.0, 'type_event': 2},
            {'time_since_start': 2.0, 'type_event': 11},
            {'time_since_start': 2.0, 'type_event': 1},
            {'time_since_start': 3.0, 'type_event': 5},  # action
        ],
        [  # 样本 3
            {'time_since_start': 1.0, 'type_event': 3},
            {'time_since_start': 1.0, 'type_event': 1},
            {'time_since_start': 1.0, 'type_event': 2},
            {'time_since_start': 1.0, 'type_event': 7},
            {'time_since_start': 2.0, 'type_event': 4},
            {'time_since_start': 2.0, 'type_event': 11},
            {'time_since_start': 2.0, 'type_event': 9},
            {'time_since_start': 3.0, 'type_event': 4},
            {'time_since_start': 3.0, 'type_event': 2},
        ]
    ]

    # 2. 创建 Dataset 和 DataLoader，num_workers=0 保证单进程
    ds = EventData(data)
    dl = DataLoader(
        ds,
        batch_size=3,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # 3. 取一个 batch 并打印各张量形状和值
    batch = next(iter(dl))
    mental_time, mental_type, action_time, action_type, pad_mask_action, action_time_to_event = batch

    #print("mental_time.shape:", mental_time.shape)
    print("mental_time:", mental_time)
    #print("mental_type.shape:", mental_type.shape)
    print("mental_type:", mental_type)

    #print("action_time.shape:", action_time.shape)
    print("action_time:", action_time)
    #print("action_type.shape:", action_type.shape)
    print("action_type:", action_type)

    #print("pad_mask_action.shape:", pad_mask_action.shape)
    print("pad_mask_action:", pad_mask_action)

    print("action_time_to_event.shape:", action_time_to_event.shape)
    print("action_time_to_event:", action_time_to_event)


if __name__ == '__main__':
    main()

