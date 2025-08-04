import numpy as np
import torch
import torch.utils.data

from model import Constants

class EventData(torch.utils.data.Dataset):
    """ Event stream dataset (with 3D mental_type). """

    def __init__(self, data):
        """
        data: list of event-streams; each stream: list of dicts with keys
        'time_since_start', 'time_since_last_event', 'type_event'.
        """
        self.mental_predicate_set = Constants.mental_predicate_set
        self.action_predicate_set = Constants.action_predicate_set

        self.mental_time = []    # List[List[float]]
        self.mental_type = []    # List[List[List[int]]]
        self.action_time = []
        self.action_type = []

        for inst in data:
            # group mental events by time
            mental_dict = {}
            for elem in inst:
                t = elem['time_since_start']
                if elem['type_event'] in self.mental_predicate_set:
                    mental_dict.setdefault(t, []).append(elem['type_event'])
            # sorted times
            times = sorted(mental_dict.keys())
            self.mental_time.append(times)
            self.mental_type.append([mental_dict[t] for t in times])

            # action remains as before
            atimes, atypes = [], []
            for elem in inst:
                if elem['type_event'] in self.action_predicate_set:
                    atimes.append(elem['time_since_start'])
                    atypes.append(elem['type_event'])
            self.action_time.append(atimes)
            self.action_type.append(atypes)

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.mental_time[idx], self.mental_type[idx], \
               self.action_time[idx], self.action_type[idx]


def padding_batch_3d(time_insts, type_insts):
    """Pad time and 3D type lists to [B, T, L]"""
    B = len(time_insts)
    max_T = max(len(seq) for seq in time_insts)
    max_L = 0
    for inst in type_insts:
        for inner in inst:
            max_L = max(max_L, len(inner))

    # pad time
    time_batch = np.array([
        seq + [Constants.PAD] * (max_T - len(seq))
        for seq in time_insts
    ], dtype=float)

    # pad types into 3D
    type_batch = np.full((B, max_T, max_L), Constants.PAD, dtype=int)
    for i, inst in enumerate(type_insts):
        for j, inner in enumerate(inst):
            type_batch[i, j, :len(inner)] = inner

    return torch.tensor(time_batch, dtype=torch.float32), torch.tensor(type_batch, dtype=torch.long)


def padding_batch(insts_time, insts_type):
    """ Pad 2D (as before) """
    max_len = max(len(inst) for inst in insts_time)
    batch_seq_time = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts_time
    ], dtype=float)
    batch_seq_type = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts_type
    ], dtype=int)
    return torch.tensor(batch_seq_time, dtype=torch.float32), torch.tensor(batch_seq_type, dtype=torch.long)


def pad_mask(padded_time, padded_type):
    return torch.where(padded_time == 0,
                       torch.zeros_like(padded_time),
                       torch.ones_like(padded_time))


def get_time_to_event(padded_time):
    diffs = torch.diff(padded_time, dim=-1)
    diffs = torch.where(diffs < 0, torch.zeros_like(diffs), diffs)
    return torch.nn.functional.pad(diffs, (1, 0), value=1e-3)


def collate_fn(insts):
    """ Collate for DataLoader. """
    # unpack
    mental_time_list, mental_type_list, action_time_list, action_type_list = zip(*insts)

    # mental: 3D types
    mental_time, mental_type = padding_batch_3d(mental_time_list, mental_type_list)
    # action: 2D as before
    action_time, action_type = padding_batch(action_time_list, action_type_list)

    pad_mask_action = pad_mask(action_time, action_type)
    action_time_to_event = get_time_to_event(action_time)

    return mental_time, mental_type, action_time, action_type, pad_mask_action, action_time_to_event


def get_dataloader(data, batch_size, shuffle=True):
    ds = EventData(data)
    return torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )

