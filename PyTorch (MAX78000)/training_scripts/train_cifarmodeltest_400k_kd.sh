#!/bin/sh
# KD training: cifarmodeltest_400k student (~400k params) + ResNet-56 teacher.
python train.py --deterministic --epochs 300 --optimizer Adam --lr 0.001 --wd 1e-4 \
    --compress policies/schedule-cifarmodeltest.yaml \
    --model cifarmodeltest_400k --dataset CIFAR10 --device MAX78000 --batch-size 128 \
    --validation-split 0 --use-bias \
    --qat-policy policies/qat_policy_cifar_test_400k.yaml \
    --kd-teacher resnet56_cifar --kd-pretrained \
    --kd-temperature 4 --kd-distill-wt 0.85 --kd-student-wt 0.15 \
    --kd-start-epoch 20 --print-freq 100 "$@"
