import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from instruct_goose.agent import Agent


def test_create_agent(agent_model):
    agent = Agent(model = agent_model)


@pytest.fixture
def tokenized_prompts(agent_tokenizer):
    prompts = [
        "Persistence is all you need?",
        "Once upon a time there's a",
        "The world is going to"
    ]

    inputs = agent_tokenizer(
        prompts,
        padding=True, truncation=True,
        return_tensors="pt",
    )
    return inputs


def test_agent_generate_text(agent_model, agent_tokenizer):
    prompts = [
        "Persistence is all you need?",
        "Once upon a time there's a",
        "The world is going to"
    ]

    inputs = agent_tokenizer(
        prompts,
        padding=True, truncation=True,
        return_tensors="pt",
    )

    output = agent_model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )

    assert output.shape[0] == len(prompts)

def test_agent_forward_pass_with_attention_mask(agent_model, tokenized_prompts):
    agent = Agent(model=agent_model)
    n_prompt = tokenized_prompts["input_ids"].shape[0]

    actions, logprobs, entropies, values = agent(
        input_ids=tokenized_prompts["input_ids"],
        attention_mask=tokenized_prompts["attention_mask"]
    )

    assert actions.shape == (n_prompt,)
    assert logprobs.shape == (3,)
    assert entropies.shape == (3,)
    assert values.shape == (3,)

def test_agent_forward_pass_without_attention_mask(agent_model, tokenized_prompts):
    # the forward pass in the RLHF trainer don't include the attention mask
    # so test the model create the attention mask itself
    agent = Agent(model=agent_model)
    n_prompt = tokenized_prompts["input_ids"].shape[0]

    actions, logprobs, entropies, values = agent(
        input_ids=tokenized_prompts["input_ids"]
    )

    assert actions.shape == (n_prompt,)
    assert logprobs.shape == (3,)
    assert entropies.shape == (3,)
    assert values.shape == (3,)


def test_agent_take_action(agent_model, agent_tokenizer):
    prompt = ["Upon once time there's a"]
    inputs = agent_tokenizer(
        prompt,
        padding=True,
        return_tensors="pt"
    )

    agent = Agent(model=agent_model)

    action, log_prob, entropy, value = agent(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    assert action.shape == (1,)
    assert value.shape == (1,)
    assert log_prob.shape == (1,)
    assert entropy.shape == (1,)