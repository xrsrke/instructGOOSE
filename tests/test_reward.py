import torch

from instruct_goose.reward import PairwiseLoss, RewardModel


def test_reward_model(default_config, reward_tokenizer, device):
    model_name = default_config["reward_model"]["model_path"]
    reward_model = RewardModel(model_name, device=device)

    prompts = ["this is suppose to be a bad text", "this is suppose to be a good text"]

    inputs = reward_tokenizer(
        prompts, padding=True, truncation=True, return_tensors="pt"
    )

    rewards = reward_model(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
    )

    assert len(rewards) == len(prompts)
    assert isinstance(rewards[0].item(), (int, float))
    assert 0 <= rewards[0].item() <= 1


def test_reward_loss():
    chosen_rewards = torch.tensor([1, 2, 3, 4])
    rejected_reward = torch.tensor([0, 1, 2, 3])

    loss_func = PairwiseLoss()
    loss = loss_func(chosen_rewards, rejected_reward)

    assert loss.numel() == 1
    assert loss.item() > 0
