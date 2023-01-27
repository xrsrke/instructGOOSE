import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from instruct_goose.agent import Agent


def test_create_agent(agent_model):
    agent = Agent(model = agent_model)


def test_agent_take_action(agent_model, agent_tokenizer):
    prompt = ["Upon once time there's a"]
    inputs = agent_tokenizer(
        prompt,
        padding=True,
        return_tensors="pt"
    )

    agent = Agent(model=agent_model)

    logits, logprobs, entropy, value = agent(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    assert logits.shape == (1, 50257)
    assert value.shape == (1,)
    assert logprobs.shape == (1, 50257)
    assert entropy.shape == (1,)