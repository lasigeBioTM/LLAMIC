#!/bin/bash
#SBATCH --job-name=MIMIC_NER_example
#SBATCH --output=MIMIC_NER_example.out
#SBATCH --gres=gpu
#SBATCH --partition=compute

task=MIMIC_example
hf auth login --token YOUR_TOKEN
MODEL_PATH_PG=llama3.1
MODEL_PATH_RC=llama3.1

datadir=data
outdir=runs/$task
mkdir -p $outdir

python3 src/llms.py --model_name_or_path_pg $MODEL_PATH_PG --model_name_or_path_rc $MODEL_PATH_RC \
  --input_file $datadir/example.csv \
  --run_mode EVAL \
  --save_chunk_size 1 --output_file $outdir/test_output.json --errors_file $outdir/test_errors.json
