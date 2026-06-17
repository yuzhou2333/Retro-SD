# -*- coding: utf-8 -*-

def process_molecules(input_file, target_file, prediction_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    targets = {}
    predictions = []

    current_target_index = None

    for line in lines:
        # 去掉行末的换行符
        line = line.strip()

        if line.startswith('T-'):
            # 提取目标分子 SMILES 和索引
            index = int(line.split('-')[1].split('\t')[0])  # 获取索引
            smi = line.split('\t')[1]  # 获取目标 SMILES
            targets[index] = smi  # 使用索引作为键
            current_target_index = index  # 更新当前目标索引

        elif line.startswith('H-'):
            # 提取预测结果 SMILES 和索引
            index = int(line.split('-')[1].split('\t')[0])  # 获取索引
            smi = line.split('\t')[-1]  # 获取预测 SMILES
            # 如果索引与当前目标相同，记录预测结果
            predictions.append((current_target_index, smi))

    # 按照索引升序排列目标
    sorted_targets = sorted(targets.items())

    # 创建一个字典来存储对应的预测结果
    predictions_dict = {}
    for target_index, pred_smi in predictions:
        if target_index not in predictions_dict:
            predictions_dict[target_index] = []
        predictions_dict[target_index].append(pred_smi)

    # 将结果写入目标文件和预测文件
    with open(target_file, 'w') as tf, open(prediction_file, 'w') as pf:
        for index, target in sorted_targets:
            tf.write(target + '\n')
            # 写入对应的预测结果，按照读取顺序
            if index in predictions_dict:
                for pred in predictions_dict[index]:
                    pf.write(pred + '\n')

def consolidate_predictions(output_file, num_files):
    with open(output_file, 'w') as out_f:
        for i in range(1, num_files + 1):
            input_file = f'prediction_{i}.txt'
            try:
                with open(input_file, 'r') as in_f:
                    lines = in_f.readlines()
                    # 将每行内容写入输出文件
                    out_f.writelines(lines)
            except FileNotFoundError:
                print(f"文件 {input_file} 不存在，跳过。")  

def consolidate_targets(output_file, num_files):
    with open(output_file, 'w') as out_f:
        for i in range(1, num_files + 1):
            input_file = f'target_{i}.txt'
            try:
                with open(input_file, 'r') as in_f:
                    lines = in_f.readlines()
                    # 将每行内容写入输出文件
                    out_f.writelines(lines)
            except FileNotFoundError:
                print(f"文件 {input_file} 不存在，跳过。")

# 调用函数，指定输入和输出文件名
for i in range(1,11):
    process_molecules(f'test_src1_tgt{i}.txt', f'target_{i}.txt', f'prediction_{i}.txt')
    
# 调用函数，指定输出文件名和文件数量
consolidate_predictions('predictions.txt', 10)
# 调用函数，指定输出文件名和文件数量
consolidate_targets('targets.txt', 10)


      
