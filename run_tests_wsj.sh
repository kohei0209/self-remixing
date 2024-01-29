#!/bin/bash

model_dir=$1
data_dir=$2  # wsj-mix dataset directory

n_epochs=5 # number of averaged checkpoints

source path.sh

# separation and evaluation on speech metrics
python evaluate_wsj.py "${model_dir}" "${data_dir}" --stage test \
    -n ${n_epochs} -c sisdr -v # -m

# asr evaluation using whisper large v2
# takes ~40min on a single RTX3090 GPU
python asr/eval_wer_wsjmix.py "${model_dir}" "${data_dir}" \
    asr/configs/whisper_asr_config.yaml
