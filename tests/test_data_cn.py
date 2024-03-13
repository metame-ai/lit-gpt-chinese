
import torch
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer


def print_data(data_file, checkpoint_dir):
    data = torch.load(data_file)
    tokenizer = Tokenizer(checkpoint_dir)
    item = data[0]
    for i in range(len(item['input_ids'][:500])):
        input_id = item['input_ids'][i]
        token = tokenizer.decode(input_id, show_special=True) 
        label = item['labels'][i]
        if "attention_mask" in item:
            mask = item['attention_mask'][i]
            print(f"{i:03d} {token:<15} {input_id:>10} {label:>10} {mask:>5}")
        else:
            print(f"{i:03d} {token:<15} {input_id:>10} {label:>10}")

        
if __name__ == "__main__":
    import fire
    fire.Fire()
        