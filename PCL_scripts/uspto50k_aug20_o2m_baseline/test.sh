#!/usr/bin/env bash

python_path=/root/miniconda3/bin/python
project_path="/root/autodl-tmp/Retro-SD"
reaction_pairs="src1-tgt1,src1-tgt2,src1-tgt3,src1-tgt4,src1-tgt5,src1-tgt6,src1-tgt7,src1-tgt8,src1-tgt9,src1-tgt10"
path_2_data=/root/autodl-tmp/Retro-SD/data_bin/uspto50k_aug20_o2m
reaction_list=${path_2_data}/reaction_list.txt

checkpoint_path=$1
checkpoint_name=checkpoint_best.pt
if [ -n "$2" ]; then
  checkpoint_name=$2
fi
model=${checkpoint_path}/${checkpoint_name}
echo "model: ${model}"
OUTPUT_DIR=$checkpoint_path

mkdir -p $OUTPUT_DIR

src="src1"
# CUDA_VISIBLE_DEVICES=0
for i in {1..10}; do
    tgt="tgt$i"
    ${python_path} ${project_path}/fairseq_cli/generate.py $path_2_data \
        --path $model \
        --task translation_multi_simple_epoch \
        --reaction-dict "$reaction_list" \
        --reaction-pairs "$reaction_pairs" \
        --gen-subset test \
        --source-lang $src \
        --target-lang $tgt \
        --encoder-langtok "tgt" \
        --scoring sacrebleu \
        --remove-bpe 'sentencepiece'\
        --batch-size 96 \
        --decoder-langtok > $OUTPUT_DIR/test_${src}_${tgt}.txt 2>&1

done

python ${project_path}/PCL_scripts/uspto50k_aug20_o2m_baseline/result_statistics.py $OUTPUT_DIR > ${OUTPUT_DIR}/result.txt 2>&1
# bash ${project_path}/experiment_scripts/uspto50k_aug20/post_test_zh.sh ${OUTPUT_DIR}/test_en_zh.txt