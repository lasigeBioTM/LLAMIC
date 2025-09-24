#!/bin/bash
#SBATCH --job-name=MIMIC_NER_example
#SBATCH --output=MIMIC_NER_example.out
#SBATCH --gres=gpu
#SBATCH --partition=compute

task=MIMIC_example
hf auth login--token YOUR_TOKEN
MODEL_PATH_NER=llama3.1 #models/llama_ner-4290
MODEL_PATH_NEL=llama3.1 #models/llama_nel-1500
MODEL_PATH_REVIEW=llama3.1

datadir=data
outdir=runs/$task
mkdir -p $outdir

python3 src/LLaMIC.py --model_name_or_path_ner $MODEL_PATH_NER --model_name_or_path_nel $MODEL_PATH_NEL --model_name_or_path_review $MODEL_PATH_REVIEW\
  --input_file $datadir/example.csv --lexicon_path $datadir/lexicon.csv --entity_type disease \
  --run_mode EVAL \
  --save_annotations yes --save_chunk_size 1 \
  --n_iterations 1 --window_size 2636 --max_input_tokens 1024 \
  --output_dir $outdir #--debug
