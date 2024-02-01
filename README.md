# Self-Remixing

Official implementation of [Self-Remixing](https://arxiv.org/abs/2211.101946), an unsupervised sound separation framework.
Self-Remixing not only works when fine-tuning pre-trained models but when training from scratch, as shown in [our paper](https://arxiv.org/abs/2309.00376).

This repository supports several single-channel sound separation methods:

- [Mixture invariant training](https://proceedings.neurips.cc/paper/2020/file/28538c394c36e4d5ea8ff5ad60562a93-Paper.pdf) (MixIT) from [Asteroid toolkit](https://github.com/asteroid-team/asteroid).
- [Efficient MixIT](https://arxiv.org/abs/2106.00847) (unofficial implementation)
- [MixIT with source sparsity loss](https://arxiv.org/abs/2106.00847) (unofficial implementation)
- [RemixIT](https://arxiv.org/abs/2110.10103) (unofficial implementation)
- [Self-Remixing](https://arxiv.org/abs/2211.10194)

# Training
This repo supports training on some public dataset.

## Environmental setup
Clone this repo
```
git clone https://github.com/kohei0209/self-remixing
```

Create anaconda environment
```
# change directory
cd self-remixing

# create environmenet and activate
conda env create -f environment.yaml
conda activate selfremixing
```

## Start training
Once creating the environmenet, training can be run as follows
```
python train.py /path/to/config /path/to/dataset
```

Currently, this repository supports training with the following public datasets.
- SMS-WSJ
- Free universal sound separation (FUSS)
- (To do) Libri2mix
- (To do) WSJ-mix used in [Self-Remixing paper](https://arxiv.org/abs/2309.00376)

Some config files for each dataset and each algorithm are prepared in ```configs/"dataset_name"/"algorithm_name"```.
For example, if you use SMS-WSJ, the command to run Self-Remixing training from scratch is
```
python train.py configs/smswsj/selfremixing/selfremixing_tfgridnet_cbs+cs_mrl1.yaml /path/to/smswsj
```

Note that we use Weights and Bias (wandb) for logging. One can change ```entity``` in the line 368 of ```train.py``` to his/her user name.


# Evaluation
When you use SMS-WSJ, evaluation can be done as follows
```
run_tests_wsj.sh /path/to/model_directory /path/to/smswsj
```
The above script first evaluates speech metrics and then do Whisper ASR evaluation.

When using FUSS, evaluation can be done as 
```
run_tests_fuss.sh /path/to/model_directory /path/to/fuss
```



# To Do
- Support Libri2Mix and WSJ-mix
- Support DDP


# LICENSE
2024 Kohei Saijo, Waseda University.

All of this code except for the code from ESPnet is released under [MIT License](https://opensource.org/license/mit/).

# Acknowledgement
This repository includes the code from [ESPnet](https://github.com/espnet/espnet) released under [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) license and the code from [Asteroid toolkit](https://github.com/asteroid-team/asteroid) released under [MIT License](https://opensource.org/license/mit/).

- ```models/conformer.py``` from ESPnet
- ```my_torch_utils/stft.py``` from ESPnet
- ```losses/mixit_wrapper.py``` from Asteroid
- ```losses/pit_wrapper.py``` from Asteroid
- ```datasets/fuss_dataset.py``` from Asteroid
- ```datasets/librimix_dataset.py``` from Asteroid

# Citations
```
@INPROCEEDINGS{saijo23_self,
  author={Saijo, Kohei and Ogawa, Tetsuji},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Self-Remixing: Unsupervised Speech Separation VIA Separation and Remixing},
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095596}
}


@inproceedings{saijo23_interspeech,
  author={Kohei Saijo and Tetsuji Ogawa},
  title={{Remixing-based Unsupervised Source Separation from Scratch}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={1678--1682},
  doi={10.21437/Interspeech.2023-1389}
}

```