from argparse import ArgumentParser

from torch import optim
from torch.utils.data import DataLoader, random_split

from transformers import AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm

from instruct_goose.reward import RewardModel, PairwiseLoss
from instruct_goose.dataset import PairDataset
from instruct_goose.utils import load_yaml

def train(config):

    MODEL_PATH = config["model"]["model_path"]
    DATA_PATH = config["data"]["data_path"]

    N_EPOCHS = config["train"]["epochs"]
    LEARNING_RATE = config["optimizer"]["lr"]
    BATCH_SIZE = config["train"]["batch_size"]

    accelerator = Accelerator()
    device = accelerator.device

    accelerator.print(config)
    accelerator.print(f"Using {accelerator.num_processes} GPUs")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    reward_model = RewardModel(model_name=MODEL_PATH, device=device)

    dataset = load_dataset(DATA_PATH, split="train")
    dataset, _ = random_split(dataset, lengths=[4, len(dataset) - 4]) # for demo purposes
    pair_dataset = PairDataset(dataset, tokenizer)
    train_dataloader = DataLoader(pair_dataset, batch_size=BATCH_SIZE)

    pairwise_loss = PairwiseLoss()
    optimizer = optim.Adam(reward_model.parameters(), lr=LEARNING_RATE)

    reward_model, optimizer, train_dataloader = accelerator.prepare(reward_model, optimizer, train_dataloader)

    reward_model.train()

    for epoch in range(N_EPOCHS):
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            # TODO: batch should return as a dict
            chosen_input_ids, chosen_attention_mask,\
            rejected_input_ids, rejected_attention_mask = batch

            chosen_rewards = reward_model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
            rejected_rewards = reward_model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)

            loss = pairwise_loss(chosen_rewards, rejected_rewards)

            accelerator.backward(loss)
            optimizer.step()

            accelerator.print(f"Epoch {epoch}, batch {batch_idx}, loss {loss.item()}")
            accelerator.gather_for_metrics({"loss": loss.detach()})

        accelerator.print(f"Epoch {epoch} finished")

    # for trackers
    # accelerator.end_training()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/train_reward.yaml", help="A pretrained transformer")
    # parser.add_argument("--data_path", type=str, default="CarperAI/openai_summarize_comparisons", help="HuggingFace dataset path")
    # parser.add_argument("--n_epochs", type=int, default=1, help="The number of epochs for training")
    # parser.add_argument("--learning_rate", type=float, default=1e-3, help="The learning rate for training")

    args = parser.parse_args()
    config = load_yaml(args.config)

    train(config=config)
