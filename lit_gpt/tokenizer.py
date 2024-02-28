# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import re
from pathlib import Path
from typing import Optional, Union

import torch


class Tokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None

        self.pad_id = 0
        self.special_token_dict = {}
        self.model_name = checkpoint_dir.name
        if "chatglm2" in self.model_name:
            # reference: https://huggingface.co/THUDM/chatglm2-6b/blob/main/tokenization_chatglm.py#L23
            # https://huggingface.co/THUDM/chatglm2-6b/blob/main/tokenization_chatglm.py#L158
            self.special_token_dict = {"[gMASK]": 64790, "sop": 64792}
        elif "baichuan2" in self.model_name:
            # "user_token_id": 195,
            # "assistant_token_id": 196,
            self.special_token_dict = {"<user>": 195, "<assistant>": 196}
        elif "chatglm3" in self.model_name:
            n_words = self.processor.vocab_size()
            role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
            special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
            for token in special_tokens:
                self.special_token_dict[token] = n_words
                n_words += 1

        self.special_token_inverse = {v: k for k, v in self.special_token_dict.items()}

        # some checkpoints have both files, `.model` takes precedence
        if (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from sentencepiece import SentencePieceProcessor

            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()

        elif (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer

            self.processor = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                bos_token = config.get("bos_token")
                self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
                eos_token = config.get("eos_token")
                self.eos_id = self.token_to_id(eos_token) if eos_token is not None else None
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
                # NOTE: some checkpoints have a list of eos tokens
                # e.g. https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat/blob/main/generation_config.json
                if isinstance(self.eos_id, list):
                    self.eos_id = self.eos_id[0]
        else:
            raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if token in self.special_token_dict:
            return self.special_token_dict[token]
        elif self.backend == "huggingface":
            id_ = self.processor.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        if not (tokenizer_config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            return False
        with open(tokenizer_config_path) as fp:
            config = json.load(fp)
        if any(config.get(check, False) for check in ("add_bos_token", "add_prefix_space")):
            return True
        # for examples that also use the Llama tokenizer, but do not have or set add_bos_token to True.
        # ex: https://huggingface.co/stabilityai/StableBeluga2/blob/main/tokenizer_config.json#L2
        return config.get("add_bos_token") is None and config.get("tokenizer_class") == "LlamaTokenizer"

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
        return_tensor: bool = True,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string).ids
        elif self.backend == "sentencepiece":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError
        if bos or (bos is None and self.use_bos):
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined a bos token")
            tokens = [bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        if return_tensor:
            return torch.tensor(tokens, dtype=torch.int, device=device)
        return tokens

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        tokens = [t for t in tokens if t not in self.special_token_dict.values()]
        return self.processor.decode(tokens)
