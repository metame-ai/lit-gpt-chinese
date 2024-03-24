import json
from abc import abstractmethod
from functools import partial
from typing import List, Dict, Union, Optional, Callable, Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, random_split

from lightning import LightningDataModule
from litgpt import Tokenizer
from litgpt.prompts import PromptStyle
from dataclasses import dataclass, field
from litgpt.data import Alpaca
from pathlib import Path


class SFTBelleDataset(Dataset):
    """An in-memory dataset for supervised finetuning with `input_ids` and `labels`.

    Args:
        data: A list of samples (dicts). The target/label must be stored under the key 'output' and the instruction
            or other data can be stored under any key as long as it is compatible with the given prompt template.
        tokenizer: The tokenizer to use. Should match the one that was used to pretrain the model.
        prompt_style: The style to apply to prompts. See `litgpt.prompts` for a list of available styles.
        max_seq_length: Truncate sequences that are longer than this value. By default, no truncation is applied.
        mask_prompt: Whether to mask the prompt section from the label (with ``ignore_index``).
        ignore_index: The index to use for elements to be ignored in the label.
        transform: An optional transform to apply to the sample before it gets tokenized. Use this to rename the
            keys in the dataset to the expected 'instruction' and 'output' keys.

    Returns a dict with two keys:
        input_ids: The encoded prompt + response
        labels: Same as input_ids, unless ``mask_prompt=True`` in which case the 'prompt' part is replaced with
            the ``ignore_index``.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: Tokenizer,
        prompt_style: Union[str, PromptStyle],
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -100,
        transform: Optional[Callable[[Any], Any]] = None,
        system_message: str = "",
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_style = (
            prompt_style if isinstance(prompt_style, PromptStyle) else PromptStyle.from_name(prompt_style)
        )
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.ignore_index = ignore_index
        self.transform = transform
        self.system_message = system_message
        self.stop_tokens = self.prompt_style.stop_tokens(tokenizer)[0]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        example = self.data[idx]
        if self.transform is not None:
            example = self.transform(example)

        input_ids, labels = [], []
        for i, msg in enumerate(example["conversations"]):
            is_first = i == 0
            bos = False if not is_first else None
            content = msg["value"].strip()
            if msg["from"] == "human":
                prompt = self.prompt_style.apply(prompt=content, is_first=is_first, 
                                                 system_message=self.system_message)
                encoded_prompt = self.tokenizer.encode(prompt, return_tensor=False, bos=bos)
                if self.mask_prompt:
                    labels += [self.ignore_index] * len(encoded_prompt)
                else:
                    labels += encoded_prompt
                input_ids += encoded_prompt
            else:
                resp = self.tokenizer.encode(content, bos=bos, return_tensor=False)
                resp += self.stop_tokens
                input_ids += resp
                labels += resp

        if self.max_seq_length > 0:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64), 
            "labels": torch.tensor(labels, dtype=torch.int64)
        }


# NOTE: sample of belle dataset
_URL = "https://raw.githubusercontent.com/baichuan-inc/Baichuan2/main/fine-tune/data/belle_chat_ramdon_10k.json"

        
@dataclass
class Belle(Alpaca):
    """Belle data module for supervised finetuning."""

    mask_prompt: bool = True
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: float = 0.1
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    prompt_style: Union[str, PromptStyle] = "chatglm3"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    download_dir: Path = Path("./data/belle")
    """The directory in which the downloaded dataset gets saved."""
    file_url: str = field(repr=False, default=_URL)
    """The URL from where to download the dataset."""
    file_name: str = field(repr=False, default="belle_sample.json")
    """The name of the dataset file to download."""
    system_message: str = ""
    """The system message to include in the prompt."""

    train_dataset: Optional[SFTBelleDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTBelleDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def setup(self, stage: str = "") -> None:
        with open(self.download_dir / self.file_name, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Partition the dataset into train and test
        train_data, test_data = random_split(
            data,
            [1.0 - self.val_split_fraction, self.val_split_fraction],
            generator=torch.Generator().manual_seed(self.seed),
        )
        train_data, test_data = list(train_data), list(test_data)

        self.train_dataset = SFTBelleDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
            system_message=self.system_message,
        )
        self.test_dataset = SFTBelleDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
            system_message=self.system_message,
        )