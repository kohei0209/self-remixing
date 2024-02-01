#!/bin/bash

model_dir=$1
data_dir=$2  # fuss dataset directory

n_epochs=5 # number of averaged checkpoints

echo "Test with Max TRF"
python evaluate_fuss.py "${model_dir}" "${data_dir}"  \
    -n ${n_epochs} -c trf --save_wavs # -v

echo "Test with Max MSi"
python evaluate_fuss.py "${model_dir}" "${data_dir}" \
    -n ${n_epochs} -c msi --save_wavs # -v
