#!/bin/bash
python finetune/lora.py \
	--precision "bf16-true" \
	--io.train_data_dir data/metame \
	--io.val_data_dir data/metame \
	--io.checkpoint_dir checkpoints/chatglm/chatglm3-6b-hf \
	--io.out_dir out/lora/metame_chatglm3_6b \
	--train.save_interval 20 \
	--train.global_batch_size 32 \
	--train.micro_batch_size 2 \
	--train.epochs 10 \
	--train.epoch_size 600 \
	--train.max_seq_length 1024 \
	--eval.interval 10 \
	--eval.max_iters 10 \
