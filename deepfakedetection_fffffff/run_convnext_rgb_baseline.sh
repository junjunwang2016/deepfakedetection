#!/usr/bin/env bash
set -euo pipefail

# ===== 基本参数（按需改） =====
NPROC=6
CFG=configs/default.yaml
OUT_DIR=runs_convnext/convnext_rgb_baseline

# ===== 数据路径（改成你的实际路径） =====
TRAIN_ROOT=/data/workspace/wangjunjun/DeepFakeDetection/Dataset/Xinye/split_train_crop
TRAIN_CSV=/data/workspace/wangjunjun/DeepFakeDetection/deepfakedetection/label/xinye_train.txt
VAL_ROOT=/data/workspace/wangjunjun/DeepFakeDetection/Dataset/Ali2024/phase1/crop_trainset/crop
VAL_CSV=/data/workspace/wangjunjun/DeepFakeDetection/Dataset/Ali2024/phase1/trainset_label.txt


# ===== 训练超参（ConvNeXt 微调常用） =====
IMG=384
BS=64                    # 每卡 batch size（显存充足可调大）
EPOCHS=20
WARMUP=2
LR_BB=5e-5
LR_HEAD=1e-3
WD=5e-2
DROPP=0.2               # drop-path

mkdir -p "$(dirname "${OUT_DIR}")"

torchrun --nproc_per_node="${NPROC}" train.py \
  --config "${CFG}" \
  --overrides \
    data.train_root="${TRAIN_ROOT}" \
    data.train_csv="${TRAIN_CSV}" \
    data.val_root="${VAL_ROOT}" \
    data.val_csv="${VAL_CSV}" \
    data.img_size=${IMG} \
    data.batch_size=${BS} \
    optim.lr_backbone=${LR_BB} \
    optim.lr_head=${LR_HEAD} \
    optim.weight_decay=${WD} \
    train.epochs=${EPOCHS} \
    train.warmup_epochs=${WARMUP} \
    model.type=rgb \
    model.backbone=convnext_base_384_in22ft1k \
    model.pretrained=true \
    model.channels_last=true \
    model.drop_path_rate=${DROPP} \
    ddp.sync_bn=false \
    aug.sbi_enabled=true \
    train.out_dir="${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}.log"
