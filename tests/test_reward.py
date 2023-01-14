import torch

from instruct_goose.reward import RewardModel, RewardLoss


def test_reward_model(default_config, reward_tokenizer):
    checkpoint = default_config["reward_model"]["model_path"]
    reward_model = RewardModel(checkpoint)

    propmt = [
        "this is suppose to be a bad text",
        "this is suppose to be a good text"
    ]

    tokenized_prompt = reward_tokenizer(propmt, return_tensors="pt")
    # input_ids = torch.unsqueeze(tokenized_prompt["input_ids"], dim=0)
    # attention_mask = torch.unsqueeze(tokenized_prompt["attention_mask"], dim=0)

    reward_scalar = reward_model(
        input_ids=tokenized_prompt["input_ids"],
        attention_mask=tokenized_prompt["attention_mask"]
    )

    assert len(reward_scalar) == len(propmt)

def test_reward_loss():
    chosen_rewards = torch.tensor([1, 2, 3, 4])
    rejected_reward = torch.tensor([0, 1, 2, 3])

    loss_func = RewardLoss()
    loss = loss_func(chosen_rewards, rejected_reward)

    assert loss.numel() == 1