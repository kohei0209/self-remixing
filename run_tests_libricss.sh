#!/bin/bash

model_dir=$1
data_dir=$2  # libri_css dataset directory.

n_epochs=5 # number of averaged checkpoints

source path.sh

libricss_root=/mnt/aoni04/saijo/libri_css

# ASR evaluation of input mixtures
for conf in espnet_asr_config whisper_asr_config; do
    if [ ! -e "${libricss_root}/asr_result/${conf}" ]; then
        echo "Start ASR evaluation of MIXTURES with ${conf}.yaml"
        python asr/eval_wer.py \
            "asr/configs/${conf}.yaml" "${libricss_root}" --eval_mixture
    else
        echo "${libricss_root}/asr_result/${conf} already exists. Skip ASR evaluation."
    fi
done


# Enhancement with pre-trained separation model
if [ ! -e "${model_dir}/enhanced" ]; then
    echo "Start enhancement"
    python asr/separation.py \
        "${model_dir}" -c wer -n ${n_epochs}
else
    echo "${model_dir}/enhanced already exists, which will be used for ASR evaluation."
fi

# ASR evaluation
for conf in espnet_asr_config whisper_asr_config; do
    if [ ! -e "${model_dir}/enhanced/asr_result/${conf}" ]; then
        echo "Start ASR evaluation with ${conf}.yaml"
        python asr/eval_wer.py \
            "asr/configs/${conf}.yaml" "${model_dir}/enhanced"
    else
        echo "${model_dir}/enhanced/asr_result/${conf} already exists. Skip ASR evaluation."
    fi
done
