from instruct_goose.reward import RewardModel


def test_reward_model(default_config, reward_tokenizer):
    checkpoint = default_config["reward_model"]["model_path"]
    reward_model = RewardModel(checkpoint)
    propmt = "this is suppose to be a bad text"
    tokenized_prompt = reward_tokenizer(propmt)

    reward_scalar = reward_model()