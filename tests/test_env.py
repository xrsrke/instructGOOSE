import torch
import pytest

from instruct_goose.env import TextEnv


@pytest.fixture
def env(tokenizer, agent_model):
    prompt = "Persistence is all you need?"
    inputs = tokenizer(prompt)
    context_length = agent_model.config.n_positions

    env = TextEnv(
        agent_model, tokenizer,
        observation_input=inputs["input_ids"],
        context_length=context_length
    )

    return env


def test_create_text_env(tokenizer, env):

    assert env.action_space.n == tokenizer.vocab_size
    assert env.action_space.n == len(env.actions)
    assert isinstance(env.actions[0], int)
    assert env.predicted_token_ids == []
    assert env.input_token_ids == []

def test_reset_env(env):
    state = env.reset()
    assert env.input_token_ids != []
    assert env.predicted_token_ids == []
    assert isinstance(state, torch.Tensor)
    assert state.ndim == 2

def test_take_action_in_text_env(env):

    # select the first action in 10th position
    # a non-eos token
    action = env.actions[10]

    env.reset()
    state, reward, terminated, truncated, info, done = env.step(action)

    assert isinstance(state, torch.Tensor)
    assert state.ndim == 2
    assert reward == 0
    assert terminated == False
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert done == False


def test_take_action_that_not_in_observation_space_in_text_env(env):
    pass