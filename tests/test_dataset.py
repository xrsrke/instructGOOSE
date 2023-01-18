import torch

from instruct_goose.dataset import PairDataset, PromptDataset


def test_create_pair_dataset(small_reward_dataset, reward_tokenizer):
    # max_length = reward_tokenizer.model_max_length
    max_length = 512
    pair_dataset = PairDataset(small_reward_dataset, reward_tokenizer, max_length)

    assert len(pair_dataset) > 0
    assert len(pair_dataset[0]) == 4
    assert isinstance(pair_dataset[0][0], torch.Tensor)
    assert isinstance(pair_dataset[0][1], torch.Tensor)
    assert isinstance(pair_dataset[0][2], torch.Tensor)
    assert isinstance(pair_dataset[0][3], torch.Tensor)

def test_prompt_dataset(small_prompt_dataset, agent_tokenizer):
    # max_length = agent_tokenizer.model_max_length
    max_length = 512
    prompt_dataset = PromptDataset(small_prompt_dataset, agent_tokenizer, max_length)

    assert len(prompt_dataset) > 0
    assert len(prompt_dataset[0]) == 2
    assert isinstance(prompt_dataset[0][0], torch.Tensor)
    assert isinstance(prompt_dataset[0][1], torch.Tensor)