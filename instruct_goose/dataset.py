# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03b_dataset.ipynb.

# %% auto 0
__all__ = ['PairDataset']

# %% ../nbs/03b_dataset.ipynb 3
from typing import Callable

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# %% ../nbs/03b_dataset.ipynb 4
class PairDataset(Dataset):
    def __init__(self, dataset: str, tokenizer: Callable, max_length: int):
        
        self.chosen = []
        self.rejected = []
        
        for data in tqdm(dataset):
            chosen, rejected = data["chosen"], data["rejected"]
            chosen_encoding = tokenizer(
                chosen,
                max_length=max_length, padding="max_length", truncation=True,
                return_tensors="pt"
            )
            rejected_encoding = tokenizer(
                rejected,
                max_length=max_length, padding="max_length", truncation=True,
                return_tensors="pt"
            )
            
            self.chosen.append({
                "input_ids": chosen_encoding["input_ids"],
                "attention_mask": chosen_encoding["attention_mask"]
            })
            self.rejected.append({
                "input_ids": rejected_encoding["input_ids"],
                "attention_mask": rejected_encoding["attention_mask"]
            })
            
    
    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, idx: int):
        return self.chosen[idx]["input_ids"],\
               self.chosen[idx]["attention_mask"],\
               self.rejected[idx]["input_ids"],\
               self.rejected[idx]["attention_mask"]