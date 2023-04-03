import torch

from instruct_goose.agent import Agent
from instruct_goose.trainer import RLHFTrainer
from instruct_goose.utils import RLHFConfig, create_reference_model


def test_create_rlhf_trainer(agent_model):
    config = RLHFConfig()

    model = Agent(agent_model)
    ref_model = create_reference_model(model)

    trainer = RLHFTrainer(model, ref_model, config)

    assert trainer.model is not None
    assert isinstance(trainer.epsilon, (int, float))


def test_compute_advantage_and_return_rlhf_trainer():
    rewards = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    advantages, returns = RLHFTrainer.compute_advantage_and_return(rewards, values)

    assert advantages.shape == (1,)
    assert returns.shape == (rewards.shape[-1],)


def test_compute_loss_rlhf_trainer(agent_model, agent_tokenizer):
    config = RLHFConfig()
    model = Agent(agent_model)
    ref_model = create_reference_model(model)
    trainer = RLHFTrainer(model, ref_model, config)

    queries = ["Are you a nice bot?", "What time is it?"]
    responses = ["Nah. I'm a mean bot.", "Nah. I'm not telling you!!!"]

    encoded_queries = agent_tokenizer(
        queries, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_responses = agent_tokenizer(
        responses, padding=True, truncation=True, return_tensors="pt"
    )
    rewards = torch.tensor([0.1, 0.9])

    loss = trainer.compute_loss(
        query_ids=encoded_queries["input_ids"],
        query_attention_mask=encoded_queries["attention_mask"],
        response_ids=encoded_responses["input_ids"],
        response_attention_mask=encoded_responses["attention_mask"],
        rewards=rewards,
    )

    assert isinstance(loss.item(), (int, float))
