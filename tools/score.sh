python score.py \
    -beam_size 10 \
    -n_best 10 \
    -augmentation 20 \
    -targets ./targets.txt \
    -predictions ./predictions.txt \
    -process_number 8 \
    -score_alpha 1 \
    -save_file ./final_results.txt
    # -detailed \
    # -source ./dataset/USPTO_50K_PtoR_aug20/test/src-test.txt \