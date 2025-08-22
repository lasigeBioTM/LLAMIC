#!/bin/bash
#SBATCH --job-name=MIMIC_NER_example
#SBATCH --output=MIMIC_NER_example.out
#SBATCH --gres=gpu
#SBATCH --partition=compute

task=MIMIC_example
huggingface-cli login --token YOUR_TOKEN

datadir=../data
outdir=$task
mkdir -p $outdir

python3 fine_tuning.py\
  --train_dataset_path $datadir/train.csv --eval_dataset_path $datadir/dev.csv \
  --llm_name llama3 --entity_type drug \
  --num_epochs 10 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --fp16 true \
  --optim adamw_bnb_8bit \
  --logging_steps 20 \
  --evaluation_strategy steps \
  --eval_steps 250 \
  --save_strategy steps \
  --save_steps 250 \
  --use_wandb true \
  --wandb_run_name llama3_finetuning_ner_diseases \
  --gradient_checkpointing true \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --max_grad_norm 0.3 \
  --push_to_hub false \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --wandb_key 10d01227e8b38c8246ee43154516d49f6c152037 \
  --output_dir $outdir --debug

