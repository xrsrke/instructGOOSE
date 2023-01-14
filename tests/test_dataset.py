import torch

from instruct_goose.dataset import PairDataset


def test_create_dataset(small_reward_dataset, tokenizer):
    max_length = tokenizer.model_max_length

    pair_dataset = PairDataset(small_reward_dataset, tokenizer, max_length)

    assert len(pair_dataset) > 0
    assert len(pair_dataset[0]) == 4
    assert isinstance(pair_dataset[0][0], torch.Tensor)
    assert isinstance(pair_dataset[0][1], torch.Tensor)
    assert isinstance(pair_dataset[0][2], torch.Tensor)
    assert isinstance(pair_dataset[0][3], torch.Tensor)
