# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import re
from pathlib import Path

import pytest
from transformers import AutoTokenizer
from transformers.models.deprecated.transfo_xl.modeling_transfo_xl import TransfoXLModelOutput
from transformers.utils import cached_file

import litgpt.config as config_module
from litgpt.tokenizer import Tokenizer



cn_model_orgs = ["Qwen", "internlm", "THUDM", "baichuan-inc", "01-ai"]
cn_model_configs = [c["name"] for c in config_module.configs if c["hf_config"]["org"] in cn_model_orgs]

@pytest.mark.parametrize(
        "config_name", cn_model_configs, ids=cn_model_configs)
def test_tokenizer_against_hf(config_name):
    access_token = os.getenv("HF_TOKEN")

    config = config_module.Config.from_name(config_name)

    repo_id = f"{config.hf_config['org']}/{config.hf_config['name']}"
    cache_dir = Path("/tmp/tokenizer_test_cache")

    # create a checkpoint directory that points to the HF files
    checkpoint_dir = cache_dir / "litgpt" / config.hf_config["org"] / config.hf_config["name"]
    if not checkpoint_dir.exists():
        file_to_cache = {}
        for file in ("tokenizer.json", "generation_config.json", "tokenizer.model", "tokenizer_config.json"):
            try:
                # download the HF tokenizer config
                hf_file = cached_file(repo_id, file, cache_dir=cache_dir / "hf", token=access_token)
            except OSError as e:
                if "gated repo" in str(e):
                    pytest.xfail("Invalid token" if access_token else "Gated repo")
                if "does not appear to have" in str(e):
                    continue
                raise e
            file_to_cache[file] = str(hf_file)
        checkpoint_dir.mkdir(parents=True)
        for file, hf_file in file_to_cache.items():
            (checkpoint_dir / file).symlink_to(hf_file)

    if re.search(r"yi-.*b", repo_id.lower()):
        # AutoTokenizer will direct to LlamaTokenizerFast
        from transformers import LlamaTokenizer
        theirs = LlamaTokenizer.from_pretrained(
            repo_id, cache_dir=cache_dir / "hf", local_files_only=True, token=access_token
        )
    else:
        try:
            theirs = AutoTokenizer.from_pretrained(
                repo_id, cache_dir=cache_dir / "hf", local_files_only=True, token=access_token,
                trust_remote_code=False,
            )
        except:
            theirs = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    ours = Tokenizer(checkpoint_dir)

    if "chatglm3" in config.name:
        assert ours.vocab_size + len(ours.special_token_dict) == theirs.vocab_size
    elif "chatglm2" in config.name:
        # https://huggingface.co/THUDM/chatglm2-6b/tree/main
        # extra 3 special tokens: "<s>", "</s>", "<unk>"
        assert ours.vocab_size + len(ours.special_token_dict) + 3 == theirs.vocab_size
    else:
        assert ours.vocab_size == theirs.vocab_size

    # assert ours.vocab_size == config.padded_vocab_size

    if "Qwen1.5" in config.name or "chatglm" in config.name:
        # even though their config defines it, it's set as None in HF
        assert isinstance(ours.bos_id, int)
        assert theirs.bos_token_id is None
    else:
        assert ours.bos_id == theirs.bos_token_id

    if config.name.startswith("stablecode"):
        # even though their config defines it, it's set as None in HF
        assert ours.eos_id == 0
        assert theirs.eos_token_id is None
    else:
        assert ours.eos_id == theirs.eos_token_id

    assert ours.pad_id == theirs.pad_token_id, f"ours: {ours.pad_id}, theirs: {theirs.pad_token_id}"

    prompt = "Hello, human!"
    special_tokens = theirs.special_tokens_map
    for _, v in special_tokens.items():
        if isinstance(v, list):
            prompt += ' ' + ' '.join(v)
        else:
            prompt += ' ' + v
    actual = ours.encode(prompt)
    expected = theirs.encode(prompt)
    if "chatglm" in config.name:
        # special_token_dict = {"[gMASK]": 64790, "sop": 64792} 
        expected = expected[2:]
    assert actual.tolist() == expected, f"special_tokens: {special_tokens}"
    assert ours.decode(actual, skip_special=True) == theirs.decode(expected, skip_special_tokens=True)


def test_tokenizer_input_validation():
    with pytest.raises(NotADirectoryError, match="The checkpoint directory does not exist"):
        Tokenizer("cocofruit")
