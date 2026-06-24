#!/bin/sh
# Knowledge-distillation: cifarmodeltest (MAX78000 student) from ResNet-56 teacher.
# Teacher weights: --kd-pretrained (NOT --pretrained; that flag is for --model only).
python train.py --deterministic --epochs 200 --optimizer Adam --lr 0.001 --wd 0 \
    --compress policies/schedule-cifarmodeltest.yaml \
    --model cifarmodeltest --dataset CIFAR10 --device MAX78000 --batch-size 256 \
    --validation-split 0 --use-bias \
    --qat-policy policies/qat_policy_cifar_test.yaml \
    --kd-teacher resnet56_cifar --kd-pretrained \
    --kd-temperature 4 --kd-distill-wt 0.7 --kd-student-wt 0.3 \
    --print-freq 100 "$@"
