"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json
import sys
from pathlib import Path

import requests
import torch
from typing import Optional, Dict
from torch.utils.data import random_split
from tqdm import tqdm
import numpy as np

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

DATA_FILE_URL = "https://raw.githubusercontent.com/baichuan-inc/Baichuan2/main/fine-tune/data/belle_chat_ramdon_10k.json"
DATA_FILE_NAME = "belle_chat_ramdon_10k.json"
DESTINATION_PATH = Path("data/belle_chat_ramdon_10k_baichuan")
CHECKPOINT_DIR = Path("checkpoints/baichuan/baichuan2-7b-chat-hf")
TEST_SPLIT_FRACTION = 0.05  # to get exactly 50 test samples
IGNORE_INDEX = -1
MASK_INPUTS = False  # as in alpaca-lora
SEED = 42


class SupervisedDataset():
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer,
        model_max_length,
        user_tokens=[195],
        assistant_tokens=[196],
    ):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = IGNORE_INDEX

    def preprocessing(self, example):
        input_ids = []
        labels = []

        for message in example["conversations"]:
            from_ = message["from"]
            value = message["value"]
            value_ids = self.tokenizer.encode(value).numpy().tolist()

            if from_ == "human":
                input_ids += self.user_tokens + value_ids
                labels += [self.tokenizer.eos_id] + [self.ignore_index] * len(
                    value_ids
                )
            else:
                input_ids += self.assistant_tokens + value_ids
                labels += [self.ignore_index] + value_ids
        input_ids.append(self.tokenizer.eos_id)
        labels.append(self.tokenizer.eos_id)
        input_ids = input_ids[: self.model_max_length]
        labels = labels[: self.model_max_length]
        input_ids += [self.tokenizer.pad_id] * (
            self.model_max_length - len(input_ids)
        )
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        input_ids = np.array(input_ids)
        labels = np.array(labels)
        # attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        attention_mask = np.not_equal(input_ids, self.tokenizer.pad_id)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int),
            "labels": torch.tensor(labels, dtype=torch.int),
            # "attention_mask": attention_mask,
        }


def prepare(
    destination_path: Path = DESTINATION_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    test_split_fraction: float = TEST_SPLIT_FRACTION,
    seed: int = SEED,
    mask_inputs: bool = MASK_INPUTS,
    data_file_name: str = DATA_FILE_NAME,
    data_file_url: str = DATA_FILE_URL,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    with open(checkpoint_dir / "lit_config.json", "r") as file:
        config = json.load(file)
        max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    data_file_path = destination_path / data_file_name
    print("Loading data file...")
    download_if_missing(data_file_path, data_file_url)
    with open(data_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    # Partition the dataset into train and test
    train_set, test_set = random_split(
        data, [1.0 - test_split_fraction, test_split_fraction], generator=torch.Generator().manual_seed(seed)
    )
    train_set, test_set = list(train_set), list(test_set)

    # DEBUG
    # train_set, test_set = train_set[:2], test_set[:1]

    dataset = SupervisedDataset(tokenizer, max_seq_length)

    print("Processing train split ...")
    res_train_set = []
    for sample in tqdm(train_set):
        res_train_set += [prepare_sample(
            example=sample,
            dataset=dataset,
        )]

    torch.save(res_train_set, destination_path / "train.pt")

    print("Processing test split ...")
    res_test_set = []
    for sample in tqdm(test_set):
        res_test_set += [prepare_sample(
            example=sample,
            dataset=dataset,
        )]
    torch.save(res_test_set, destination_path / "test.pt")

    print(f"train has {len(res_train_set):,} samples")
    print(f"test has {len(res_test_set):,} samples")


def download_if_missing(file_path: Path, file_url: str):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)


def prepare_sample(
    example: dict,
    dataset: SupervisedDataset,
):
    """Processes a single sample.

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    return dataset.preprocessing(example)


def generate_prompt(example, idx):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    '''
    {
        "id": "77771",
        "conversations": [
        {
            "from": "human",
            "value": "写一篇关于气候变化对海洋生态的影响的文章。\n"
        },
        {
            "from": "gpt",
            "value": "好的，以下是你要求的文章：\n气候变化对海洋生态的影响\n气候变化对海洋生态产生了广泛的影响。全球变暖导致海洋温度上升，这可能会对许多水生生物的生存、繁殖和迁移造成巨大的影响。另外，海洋酸化也是一个问题，因为大气中二氧化碳的增加导致海水中的pH值下降，使得许多生物的外骨骼变得脆弱，从而影响它们的生存和繁殖。\n此外，气候变化还会引起海平面上升，这不仅对海岸线和沿海居民产生影响，还会对许多海洋生物的栖息地造成影响。一些岛屿上的陆地生物面临几乎完全消失的风险，而另一些生物范围则会扩大到更广泛的领域，这可能会导致新的竞争和生态平衡的改变。\n总之，气候变化并不只是一个地球气温的问题，它对海洋生态系统产生了广泛和深远的影响。\n"
        },
        {
            "from": "human",
            "value": "从刚刚的文章中提取出气候变化对海洋生物繁殖的影响。\n"
        },
        {
            "from": "gpt",
            "value": "文章已经提到过气候变化会对海洋生物的生存、繁殖和迁移造成影响。例如，海洋酸化会使得许多生物的外骨骼变得脆弱，进而影响它们的生存和繁殖。此外，海洋温度上升也可能会对很多生物的生存和繁殖产生负面影响。"
        }
        ]
    },
    '''
    pass


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
