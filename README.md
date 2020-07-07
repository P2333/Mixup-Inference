# Mixup Inference

This repository contains the codes for reproducing most of the results of our paper.

Paper Tittle:

[Mixup Inference: Better Exploiting Mixup to Defend Adversarial Attacks](https://openreview.net/forum?id=ByxtC2VtPB&noteId=ByxtC2VtPB) (ICLR 2020)

Tianyu Pang*, Kun Xu* and Jun Zhu

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 16.04.3
- GPU: Geforce 1080 Ti or Tesla P100
- Cuda: 10.0, Cudnn: v7.4
- Python: 3.5.2
- PyTorch: 1.2.0
- Torchvision: 0.4.0

For convenience, we provide the *requ.txt* file to install the virtualenv that is sufficient run the codes.

In the following, we first provide the codes for training with [mixup](https://arxiv.org/pdf/1710.09412.pdf) and [interpolated AT](https://arxiv.org/pdf/1906.06784.pdf). After that, the evaluation codes, such as attacking our mixup inference (MI) method and other baselines, are provided.

## Training codes

### Training with the mixup mechinism

Let `dataset` be `cifar10` or `cifar100`, the command for training models with the mixup mechanism is
```shell
python train_resnet_mixup.py -model resnet50 -lr 0.01 -adv_ratio 0. -mixup_alpha 1. -data [dataset] -bs 64
```
When applying mixup, the initial learning rate is `0.01`, the alpha is `1.0`, the optimizer is `mom` and we use the `ResNet-50` architecture proposed by [He et al. (2016)](https://arxiv.org/abs/1603.05027). The training epoch on both CIFAR-10 and CIFAR-100 is set as 200. Pretrained models are avaiable: [mixup checkpoint (CIFAR-10)](http://ml.cs.tsinghua.edu.cn/~tianyu/MixupInference/model_checkpoints/CIFAR10/model_mixup.ckpt) and [mixup checkpoint (CIFAR-100)](http://ml.cs.tsinghua.edu.cn/~tianyu/MixupInference/model_checkpoints/CIFAR100/model_mixup.ckpt).

### Training with the interpolated AT mechinism

The command for training models with the interpolated AT mechanism is
```shell
python train_resnet_mixup.py -model resnet50 -lr 0.1 -adv_ratio 0.5 -mixup_alpha 1. -data [dataset] -bs 64
```
When applying interpolated AT, the initial learning rate is `0.1`, the alpha is `1.0`, the optimizer is `mom`. The ratio between the clean samples and the adversarial ones is 1:1. Pretrained models are avaiable: [IAT checkpoint (CIFAR-10)](http://ml.cs.tsinghua.edu.cn/~tianyu/MixupInference/model_checkpoints/CIFAR10/model_IAT.ckpt) and [IAT checkpoint (CIFAR-100)](http://ml.cs.tsinghua.edu.cn/~tianyu/MixupInference/model_checkpoints/CIFAR100/model_IAT.ckpt).


## Evaluation codes

We mainly evluate the PGD attacks for different inference-phase defenses. Here the `eps` is by default `8/255`, the pixels are normalized to the interval `[-1,1]`. **About the detailed parameter settings to re-implement the results in our paper, please refer to Table 4 and Table 5 in our appendix.**

### Evaluating MI-PL

Let `dataset` be `cifar10` or `cifar100`. The `model_checkpoint` be the file of trained model checkpoint, which could be trained by mixup or interpolated AT or other training methods. The evaluation command is
```shell
python attack_resnet_mixuptest_PL.py -targeted=False -nbiter=10 -data=[dataset] -model=resnet50 -oldmodel=model_checkpoint/model.ckpt -lamda=0.5
```
The FLAG `targeted` indicates whether use targeted attacks or untargeted attacks. `nbiter` is the iterations steps of the PGD attacks. For exmaple, the attack here is `untargeted PGD-10`. Here `lamda` is the mixup ratio for MI-PL.

### Evaluating MI-OL

Similar to the command for MI-PL, the one for evaluating MI-OL is
```shell
python attack_resnet_mixuptest_OL.py -targeted=False -nbiter=10 -data=[dataset] -model=resnet50 -oldmodel=model_checkpoint/model.ckpt -lamda=0.5
```
Here `lamda` is the mixup ratio for MI-OL, where we use `lamda=0.5` for mixup+MI-OL and `lamda=0.6` for IAT+MI-OL.

### Evaluating MI-Combined

The command is
```shell
python attack_resnet_mixuptest_Combined.py -targeted=False -data=[dataset] -model=resnet50 -oldmodel=model_checkpoint/model.ckpt -lamdaPL=0.5 -lamdaOL=0.4 -threshold=0.2
```
Here `lamdaPL` is the lambda for MI-PL, `lamdaOL` is the lambda for MI-OL, `threshold` is used to decide whether the input is adversarial according to the detection result returned by MI-PL. 

### Evaluating Baselines

The command for evaluating `Gaussian noise` baseline is
```shell
python -u attack_resnet_baselines.py -targeted=False -nbiter=10 -data=[dataset] -eps=8 -model=resnet50 -oldmodel=model_checkpoint/model.ckpt -baseline='gaussian' -num_sample=50 -sigma=0.05 -numtest=1000
```

The command for evaluating `Rotation` baseline is
```shell
python -u attack_resnet_baselines.py -targeted=False -nbiter=10 -data=[dataset] -eps=8 -model=resnet50 -oldmodel=model_checkpoint/model.ckpt -baseline='Rotation' -num_sample=50 -Rotation=20 -numtest=1000
```

The command for evaluating `Xie et al. (2018)` baseline is
```shell
python -u attack_resnet_baselines.py -targeted=False -nbiter=10 -data=[dataset] -eps=8 -model=resnet50 -oldmodel=model_checkpoint/model.ckpt -baseline='Xie' -num_sample=50 -Xielower=24 -Xieupper=32 -numtest=1000
```

The command for evaluating `Guo et al. (2018)` baseline is
```shell
python -u attack_resnet_baselines.py -targeted=False -nbiter=10 -data=[dataset] -eps=8 -model=resnet50 -oldmodel=model_checkpoint/model.ckpt -baseline='Guo' -num_sample=50 -Guolower=24 -Guoupper=32 -numtest=1000
```

### Evaluating Adaptive Attacks for MI-OL

The adaptive attacks are perform under different number of adaptive samples, as shown in the command below
```shell
x=( 1 2 3 4 5 10 15 20 25 30 )
for num_sample in ${x[@]}
do
    python attack_resnet_mixuptest_OL_adaptive.py -targeted=False -nbiter=10 -data=[dataset] -model=resnet50 \
    -oldmodel=model_checkpoint/model.ckpt \
    -lamda=0.5 -adaptive=True -adaptive_num=$num_sample;
done
```
This is an example of adaptive PGD-10 attack.
