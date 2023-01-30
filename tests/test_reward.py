import torch

from instruct_goose.reward import RewardModel, RewardLoss


def test_reward_model(default_config, reward_tokenizer):
    checkpoint = default_config["reward_model"]["model_path"]
    reward_model = RewardModel(checkpoint)

    prompts = [
        "this is suppose to be a bad text",
        "this is suppose to be a good text"
    ]

    rewards = reward_model(prompts=prompts)

    assert len(rewards) == len(prompts)
    assert isinstance(rewards[0].item(), (int, float))
    assert 0 <= rewards[0].item() <= 1

def test_reward_loss():
    chosen_rewards = torch.tensor([1, 2, 3, 4])
    rejected_reward = torch.tensor([0, 1, 2, 3])

    loss_func = RewardLoss()
    loss = loss_func(chosen_rewards, rejected_reward)

    assert loss.numel() == 1