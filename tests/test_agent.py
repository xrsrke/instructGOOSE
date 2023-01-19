from transformers import AutoModelForCausalLM, AutoTokenizer

from instruct_goose.agent import Agent


def test_agent(agent_model, agent_tokenizer):
    prompts = [
        "Upon once time there's a",
        "the world is going to",
        "what's up everybody"
    ]

    inputs = agent_tokenizer(
        prompts,
        padding=True, truncation=True, return_tensors="pt"
    )

    agent = Agent(agent_model)
    logits, logprobs, entropy, value = agent(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    assert logits.shape == (3, 50257)
    assert value.shape == (3,)
    assert logprobs.shape == (3, 50257)
    assert entropy.shape == (3,)