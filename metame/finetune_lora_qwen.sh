#!/bin/bash
python finetune/lora.py \
	--precision "bf16-true" \
	--io.train_data_dir data/metame_qwen \
	--io.val_data_dir data/metame_qwen \
	--io.checkpoint_dir checkpoints/qwen/Qwen1.5-7B-Chat \
	--io.out_dir out/lora/metame_qwen15_7b \
	--train.save_interval 20 \
	--train.global_batch_size 32 \
	--train.micro_batch_size 2 \
	--train.epochs 5 \
	--train.epoch_size 600 \
	--train.max_seq_length 1024 \
	--eval.interval 10 \
	--eval.max_iters 10 \
