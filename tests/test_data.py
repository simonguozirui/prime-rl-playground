import torch
from zeroband.data import collate_causal_mask


def test_collate_fn():
    tensors = [[0, 1, 2, 3, 4], [0, 0, 3, 4, 1, 7]]

    batch = [{"input_ids": torch.Tensor(tensor)} for tensor in tensors]

    collate_fn = collate_causal_mask(max_seq_length=4)
    collated = collate_fn(batch)

    assert collated is not None

    assert collated["input_ids"][0].tolist() == [0, 1, 2, 3]
    assert collated["labels"][0].tolist() == [1, 2, 3, 4]

    assert collated["input_ids"][1].tolist() == [0, 0, 3, 4]
    assert collated["labels"][1].tolist() == [0, 3, 4, 1]
