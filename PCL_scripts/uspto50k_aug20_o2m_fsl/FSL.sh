#!/usr/bin/env bash

# 控制是否启用wandb
USE_WANDB=false  # 设置为false可以禁用wandb

# wandb相关配置
if [ "$USE_WANDB" = true ]; then
    export WANDB_API_KEY=""  # 替换为您的wandb API密钥
    export WANDB_NAME=""  # 实验名称
    export WANDB_TAGS=""  # 可选：添加标签
    WANDB_ARGS="--use-wandb --wandb-project your-project-name --wandb-entity your-entity-name" # 对your-project-name与your-entity-name修改为您的实体与名称
else
    WANDB_ARGS=""
fi

python_path=/root/miniconda3/bin/python
project_path="/root/autodl-tmp/Retro-SD"
reaction_pairs="src1-tgt1,src1-tgt2,src1-tgt3,src1-tgt4,src1-tgt5,src1-tgt6,src1-tgt7,src1-tgt8,src1-tgt9,src1-tgt10"
path_2_data=/root/autodl-tmp/Retro-SD/data_bin/uspto50k_aug20_o2m_fsl
reaction_list=${path_2_data}/reaction_list.txt

SAVE_DIR=${project_path}/checkpoints/uspto50k_aug20_o2m/FSL
mkdir -vp $SAVE_DIR

CUDA_VISIBLE_DEVICES=0
${python_path} ${project_path}/fairseq_cli/train.py $path_2_data \
  --save-dir $SAVE_DIR \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer_retro --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 5 \
  --decoder-langtok \
  --encoder-langtok "tgt" \
  --reaction-dict "$reaction_list" \
  --reaction-pairs "$reaction_pairs" \
  --label-smoothing 0.0 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 0.003 \
  --share-decoder-input-output-embed \
  --max-epoch 157 \
  --dropout 0.3 --attention-dropout 0.3 \
  --weight-decay 0.0 \
  --max-tokens 8192 --update-freq 2 \
  --save-interval 1 \
  --seed 222 --log-format simple --log-interval 10 \
  --bpe sentencepiece \
  --pure-batch \
  --RS-epoch \
  --online-distillation-MNMT \
  --reaction-aware-online-distillation \
  --criterion label_smoothed_cross_entropy_with_online_distillation \
  --online-distillation-weight 0.8 \
  --selective-online-distillation "soft" \
  --selective-online-distillation-level sentence \
  $WANDB_ARGS > ${SAVE_DIR}/train.log 2>&1


bash ${project_path}/PCL_scripts/uspto50k_aug20_o2m_fsl/test.sh $SAVE_DIR
