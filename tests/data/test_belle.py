
import torch
import sys
import json
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from litgpt import Tokenizer
from litgpt.data import SFTBelleDataset, Belle


def test_dataset(json_file, checkpoint_dir, prompt_style, sys_msg=""):
    with open(json_file, 'r') as f:
        data = json.load(f)

    tokenizer = Tokenizer(checkpoint_dir)
    dataset = SFTBelleDataset(data, tokenizer, prompt_style, 
                              system_message=sys_msg)
    item = dataset.__getitem__(0)
    for i in range(len(item['input_ids'][:100])):
        input_id = item['input_ids'][i]
        token = tokenizer.decode(input_id)
        label = item['labels'][i]
        print(f"{i:03d} {token:<15} {input_id:>10} {label:>10}")


def test_loader(json_file, checkpoint_dir, prompt_style, sys_msg=""):

    data_path = Path(json_file)
    tokenizer = Tokenizer(checkpoint_dir)
    belle = Belle(val_split_fraction=0.5, download_dir=data_path.parent, 
                  file_name=data_path.name, num_workers=0,
                  prompt_style=prompt_style, system_message=sys_msg)
    belle.connect(tokenizer, batch_size=2, max_seq_length=10)
    belle.prepare_data()
    belle.setup()

    train_dataloader = belle.train_dataloader()
    val_dataloader = belle.val_dataloader()

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))

    assert train_batch.keys() == val_batch.keys() == {"input_ids", "labels"}
    assert all(seq.shape == (2, 10) for seq in train_batch.values())
    assert all(seq.shape == (2, 10) for seq in val_batch.values())


if __name__ == "__main__":
    import fire
    fire.Fire()

