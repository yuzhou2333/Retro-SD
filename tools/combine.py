# -*- coding: utf-8 -*-

def consolidate_predictions(output_file, num_classes):
    with open(output_file, 'w') as out_f:
        for i in num_classes:
            input_file = f'prediction_{i}.txt'
            try:
                with open(input_file, 'r') as in_f:
                    lines = in_f.readlines()
                    # 将每行内容写入输出文件
                    out_f.writelines(lines)
            except FileNotFoundError:
                print(f"文件 {input_file} 不存在，跳过。")  

def consolidate_targets(output_file, num_classes):
    with open(output_file, 'w') as out_f:
        for i in num_classes:
            input_file = f'target_{i}.txt'
            try:
                with open(input_file, 'r') as in_f:
                    lines = in_f.readlines()
                    # 将每行内容写入输出文件
                    out_f.writelines(lines)
            except FileNotFoundError:
                print(f"文件 {input_file} 不存在，跳过。")

num_classes1 = [1, 2, 6, 3, 7]
num_classes2 = [9, 4, 8, 5, 10]
num_classes3 = [6, 7, 8, 9, 10]
    
# 调用函数，指定输出文件名和文件数量
consolidate_predictions('predictions_high.txt', num_classes1)
# 调用函数，指定输出文件名和文件数量
consolidate_targets('targets_high.txt', num_classes1)

# 调用函数，指定输出文件名和文件数量
consolidate_predictions('predictions_low.txt', num_classes2)
# 调用函数，指定输出文件名和文件数量
consolidate_targets('targets_low.txt', num_classes2)

# 调用函数，指定输出文件名和文件数量
consolidate_predictions('predictions_6_10.txt', num_classes3)
# 调用函数，指定输出文件名和文件数量
consolidate_targets('targets_6_10.txt', num_classes3)