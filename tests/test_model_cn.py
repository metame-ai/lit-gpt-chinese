import sys
import json
import os
from functools import partial
from pathlib import Path
from urllib.request import urlretrieve

import pytest
import torch
from conftest import RunIf
from lightning import Fabric
import lightning 
from lightning.fabric.utilities.imports import _IS_WINDOWS, _TORCH_GREATER_EQUAL_2_2

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.config as config_module

@torch.inference_mode()
@pytest.mark.parametrize(
    "ours_kwargs",
    [
        {"name": "internlm2-chat-1_8b"}, {"name": "internlm2-chat-7b"},
        {"name": "internlm2-chat-20b"},
    ],
)
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_interlm2(ours_kwargs, device, dtype):
    import importlib
    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_internlm2

    torch.set_default_dtype(dtype)

    T = 5
    ours_config = Config.from_name(
        padded_vocab_size=10000, n_layer=2, n_head=8, n_embd=32, 
        intermediate_size=86, n_query_groups=2, **ours_kwargs
    )

    hf_name_ori = ours_config.hf_config["name"]
    hf_name = hf_name_ori.replace("-", "_")
    workdir = wd / "tests" / "reference_models" / hf_name
    workdir.mkdir(parents=True, exist_ok=True)
    file_paths = [workdir / f"configuration_internlm2.py", 
                  workdir / f"modeling_internlm2.py", 
                  workdir / f"config.json"]
    urls = [
        f"https://huggingface.co/internlm/{hf_name_ori}/raw/main/configuration_internlm2.py",
        f"https://huggingface.co/internlm/{hf_name_ori}/raw/main/modeling_internlm2.py",
        f"https://huggingface.co/internlm/{hf_name_ori}/raw/main/config.json",
    ]
    for file_path, url in zip(file_paths, urls):
        if not file_path.is_file():
            urlretrieve(url=url, filename=file_path)
    
    configuration = importlib.import_module(
        f"reference_models.{hf_name}.configuration_internlm2")   
    modeling = importlib.import_module(
        f"reference_models.{hf_name}.modeling_internlm2")
    
    hf_config_file = workdir / f"config.json"
    with open(hf_config_file, "r") as f:
        hf_config_dict = json.load(f)
    hf_config_dict.update(
        {
            "vocab_size": ours_config.padded_vocab_size, 
            "num_hidden_layers": ours_config.n_layer,
            "num_attention_heads": ours_config.n_head,
            "num_key_value_heads": ours_config.n_query_groups,
            "hidden_size": ours_config.n_embd,
            "intermediate_size": ours_config.intermediate_size,
            "torch_dtype": dtype,
        }
    )
    theirs_config = configuration.InternLM2Config(
        **hf_config_dict
    )

    theirs_model = modeling.InternLM2ForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_internlm2(ours_config, state_dict, theirs_state_dict, verbose=False)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)

@torch.inference_mode()
@pytest.mark.parametrize(
    "ours_kwargs",
    [{"name": "Qwen1.5-1.8B-Chat"}, {"name": "Qwen1.5-0.5B-Chat"},
     {"name": "Qwen1.5-7B-Chat"}, {"name": "Qwen1.5-4B-Chat"}, 
     {"name": "Qwen1.5-14B-Chat"}, {"name": "Qwen1.5-72B-Chat", "n_query_groups": 2}],
)
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_qwen1_5(ours_kwargs, device, dtype):
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_qwen2

    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        padded_vocab_size=10000, n_layer=2, n_head=8, 
        n_embd=32, intermediate_size=86, **ours_kwargs
    )
    T = 5
    repo_id = f'{ours_config.hf_config["org"]}/{ours_config.hf_config["name"]}'
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    config_path = f"/tmp/{repo_id.split('/')[-1]}.json"
    if not os.path.isfile(config_path):
        urlretrieve(url=url, filename=config_path)

    with open(config_path, "r") as f:
        hf_config_dict = json.load(f)
    hf_config_dict.update(
        {
            "vocab_size": ours_config.padded_vocab_size, 
            "num_hidden_layers": ours_config.n_layer,
            "num_attention_heads": ours_config.n_head,
            "hidden_size": ours_config.n_embd,
            "intermediate_size": ours_config.intermediate_size,
            "num_key_value_heads": ours_config.n_query_groups,
            "torch_dtype": dtype,
        }
    )

    theirs_config = Qwen2Config(**hf_config_dict)
    theirs_model = Qwen2ForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_qwen2(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    "ours_kwargs",
    [
        {"name": "baichuan2-7b-chat-hf"}, {"name": "baichuan2-7b-base-hf"},
    ],
)
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (
            torch.device("cpu"), torch.float32,
        ),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=(AssertionError, ImportError), strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_baichuan7b(ours_kwargs, device, dtype):
    import importlib
    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_baichuan2
    import transformers

    torch.set_default_dtype(dtype)

    T = 5
    ours_config = Config.from_name(
        padded_vocab_size=10000, n_layer=2, n_head=8, n_embd=32, 
        intermediate_size=86, **ours_kwargs
    )

    hf_name_ori = ours_config.hf_config["name"].lower()
    hf_name = hf_name_ori.replace("-", "_")
    workdir = wd / "tests" / "reference_models" / hf_name
    workdir.mkdir(parents=True, exist_ok=True)
    file_paths = [workdir / f"configuration_baichuan.py", 
                  workdir / f"modeling_baichuan.py", 
                  workdir / f"config.json",
                  workdir / f"generation_utils.py"]
    urls = [
        f"https://huggingface.co/baichuan-inc/{hf_name_ori}/raw/main/configuration_baichuan.py",
        f"https://huggingface.co/baichuan-inc/{hf_name_ori}/raw/main/modeling_baichuan.py",
        f"https://huggingface.co/baichuan-inc/{hf_name_ori}/raw/main/config.json",
        f"https://huggingface.co/baichuan-inc/{hf_name_ori}/raw/main/generation_utils.py",
    ]
    for file_path, url in zip(file_paths, urls):
        if not file_path.is_file():
            urlretrieve(url=url, filename=file_path)

    configuration = importlib.import_module(
        f"reference_models.{hf_name}.configuration_baichuan")   
    modeling = importlib.import_module(
        f"reference_models.{hf_name}.modeling_baichuan")
    
    hf_config_file = workdir / f"config.json"
    with open(hf_config_file, "r") as f:
        hf_config_dict = json.load(f)
    hf_config_dict.update(
        {
            "vocab_size": ours_config.padded_vocab_size, 
            "num_hidden_layers": ours_config.n_layer,
            "num_attention_heads": ours_config.n_head,
            "hidden_size": ours_config.n_embd,
            "intermediate_size": ours_config.intermediate_size,
            "torch_dtype": dtype,
        }
    )
    theirs_config = configuration.BaichuanConfig(
        **hf_config_dict
    )

    theirs_model = modeling.BaichuanForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_baichuan2(ours_config, state_dict, theirs_state_dict, verbose=False)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    "ours_kwargs",
    [
        # NOTE: transformers==4.29.2 is required for 13b models
        {"name": "baichuan2-13b-chat-hf"}, {"name": "baichuan2-13b-base-hf"},
    ],
)
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        pytest.param(
            torch.device("cpu"), torch.float32,
            marks=[pytest.mark.xfail(raises=ImportError, strict=False)],
        ),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=(AssertionError, ImportError), strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_baichuan13b(ours_kwargs, device, dtype):
    import importlib
    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_baichuan2
    import transformers

    torch.set_default_dtype(dtype)

    T = 5
    ours_config = Config.from_name(
        padded_vocab_size=10000, n_layer=2, n_head=8, n_embd=32, 
        intermediate_size=86, **ours_kwargs
    )

    hf_name_ori = ours_config.hf_config["name"].lower()
    hf_name = hf_name_ori.replace("-", "_")
    workdir = wd / "tests" / "reference_models" / hf_name
    workdir.mkdir(parents=True, exist_ok=True)
    file_paths = [workdir / f"configuration_baichuan.py", 
                  workdir / f"modeling_baichuan.py", 
                  workdir / f"config.json",
                  workdir / f"generation_utils.py"]
    urls = [
        f"https://huggingface.co/baichuan-inc/{hf_name_ori}/raw/main/configuration_baichuan.py",
        f"https://huggingface.co/baichuan-inc/{hf_name_ori}/raw/main/modeling_baichuan.py",
        f"https://huggingface.co/baichuan-inc/{hf_name_ori}/raw/main/config.json",
        f"https://huggingface.co/baichuan-inc/{hf_name_ori}/raw/main/generation_utils.py",
    ]
    for file_path, url in zip(file_paths, urls):
        if not file_path.is_file():
            urlretrieve(url=url, filename=file_path)

    if '-13b' in ours_kwargs['name'] and transformers.__version__ != "4.29.2":
        raise ImportError("transformers==4.29.2 is required for 13b models")
    
    configuration = importlib.import_module(
        f"reference_models.{hf_name}.configuration_baichuan")   
    modeling = importlib.import_module(
        f"reference_models.{hf_name}.modeling_baichuan")
    
    hf_config_file = workdir / f"config.json"
    with open(hf_config_file, "r") as f:
        hf_config_dict = json.load(f)
    hf_config_dict.update(
        {
            "vocab_size": ours_config.padded_vocab_size, 
            "num_hidden_layers": ours_config.n_layer,
            "num_attention_heads": ours_config.n_head,
            "hidden_size": ours_config.n_embd,
            "intermediate_size": ours_config.intermediate_size,
            "torch_dtype": dtype,
        }
    )
    theirs_config = configuration.BaichuanConfig(
        **hf_config_dict
    )

    theirs_model = modeling.BaichuanForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    if '-13b' in ours_kwargs['name']:
        # explicitly initialize the weights to random values
        for k, v in theirs_state_dict.items():
            theirs_state_dict[k] = torch.empty_like(v, dtype=dtype, device=device).normal_(mean=0, std=1)
        theirs_model.load_state_dict(theirs_state_dict)
    state_dict = {}
    copy_weights_hf_baichuan2(ours_config, state_dict, theirs_state_dict, verbose=False)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    "ours_kwargs",
    [
        {"name": "chatglm3-6b-hf"}, {"name": "chatglm3-6b-32k-hf"}, {"name": "chatglm3-6b-base-hf"},
        {"name": "chatglm2-6b-hf"}, 
    ],
)
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_chatglm(ours_kwargs, device, dtype):
    import importlib
    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_chatglm2

    torch.set_default_dtype(dtype)

    T = 5
    ours_config = Config.from_name(
        padded_vocab_size=10000, n_layer=2, n_head=8, n_embd=32, 
        intermediate_size=86, n_query_groups=2, **ours_kwargs
    )

    hf_name_ori = ours_config.hf_config["name"].lower()
    hf_name = hf_name_ori.replace("-", "_")
    workdir = wd / "tests" / "reference_models" / hf_name
    workdir.mkdir(parents=True, exist_ok=True)
    file_paths = [workdir / f"configuration_chatglm.py", 
                  workdir / f"modeling_chatglm.py", 
                  workdir / f"config.json"]
    urls = [
        f"https://huggingface.co/THUDM/{hf_name_ori}/raw/main/configuration_chatglm.py",
        f"https://huggingface.co/THUDM/{hf_name_ori}/raw/main/modeling_chatglm.py",
        f"https://huggingface.co/THUDM/{hf_name_ori}/raw/main/config.json",
    ]
    for file_path, url in zip(file_paths, urls):
        if not file_path.is_file():
            urlretrieve(url=url, filename=file_path)
    
    configuration_chatglm = importlib.import_module(
        f"reference_models.{hf_name}.configuration_chatglm")   
    modeling_chatglm = importlib.import_module(
        f"reference_models.{hf_name}.modeling_chatglm")
    
    hf_config_file = workdir / f"config.json"
    with open(hf_config_file, "r") as f:
        hf_config_dict = json.load(f)
    hf_config_dict.update(
        {
            "padded_vocab_size": ours_config.padded_vocab_size, 
            "num_layers": ours_config.n_layer,
            "num_attention_heads": ours_config.n_head,
            "multi_query_group_num": ours_config.n_query_groups,
            "kv_channels": ours_config.n_embd // ours_config.n_head,
            "hidden_size": ours_config.n_embd,
            "ffn_hidden_size": ours_config.intermediate_size,
            "torch_dtype": dtype,
        }
    )
    theirs_config = configuration_chatglm.ChatGLMConfig(
        **hf_config_dict
    )
    assert ours_config.intermediate_size == theirs_config.ffn_hidden_size, \
        (f"ours_config.intermediate_size: {ours_config.intermediate_size}, "
         f"theirs_config.ffn_hidden_size: {theirs_config.ffn_hidden_size}")

    theirs_model = modeling_chatglm.ChatGLMForConditionalGeneration(theirs_config, empty_init=False).to(device)
    theirs_state_dict = theirs_model.state_dict()
    # explicitly initialize the weights to random values
    for k, v in theirs_state_dict.items():
        theirs_state_dict[k] = torch.empty_like(v, dtype=dtype, device=device).normal_(mean=0, std=1)
    theirs_model.load_state_dict(theirs_state_dict)
    state_dict = {}
    copy_weights_hf_chatglm2(ours_config, state_dict, theirs_state_dict, verbose=False)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    "ours_kwargs",
    [{"name": "yi-6b-chat-hf"}, {"name": "yi-6b-200k-hf"}, {"name": "yi-6b-hf"},
     {"name": "yi-34b-chat-hf"}, {"name": "yi-34b-200k-hf"}, {"name": "yi-34b-hf"},
     {"name": "yi-9b-hf"}
    ],
)
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_yi(ours_kwargs, device, dtype):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        padded_vocab_size=10000, n_layer=2, n_head=8, n_embd=32, intermediate_size=86, **ours_kwargs
    )
    T = 5
    theirs_config = LlamaConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = LlamaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)