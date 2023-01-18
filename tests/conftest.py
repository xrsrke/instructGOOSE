import os

import pytest
from torch.utils.data import random_split
from transformers import AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()

from instruct_goose.utils import load_yaml



@pytest.fixture
def default_config():
    # config_path = Path()
    # package_path = os.path.abspath(os.path.dirname(__file__))
    package_path = os.environ.get('PACKAGE_PATH')
    return load_yaml(package_path + "/configs/sentiment_config.yml")

@pytest.fixture
def tokenizer(default_config):
    tokenizer_path = default_config["model"]["tokenizer_path"]
    return AutoTokenizer.from_pretrained(tokenizer_path)


#### REWARD MODEL

@pytest.fixture
def reward_dataset(default_config):
    dataset_checkpoint = default_config["reward_data"]["data_path"]
    dataset = load_dataset(dataset_checkpoint)
    return dataset

@pytest.fixture
def small_reward_dataset(reward_dataset):
    small_dataset, _ = random_split(reward_dataset["train"], [10, len(reward_dataset["train"]) - 10])
    return small_dataset

@pytest.fixture
def reward_tokenizer(default_config):
    reward_checkpoint = default_config["reward_model"]["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(reward_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


#### RL MODEL

@pytest.fixture
def small_prompt_dataset(default_config):
    dataset_checkpoint = default_config["agent_data"]["data_path"]
    dataset = load_dataset(dataset_checkpoint)
    small_dataset, _ = random_split(dataset["train"], [10, len(dataset["train"]) - 10])

    return small_dataset

@pytest.fixture
def agent_tokenizer(default_config):
    agent_checkpoint = default_config["model"]["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(agent_checkpoint)
    # tokenizer.pad_token = tokenizer.eos_token
    return tokenizer