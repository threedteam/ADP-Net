import os
import re

folder_path = r"./train/img"
output_txt = r"train_img.txt"

def extract_last_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else float('inf')

file_list = [os.path.abspath(os.path.join(folder_path, f))
             for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f))]

# file_list.sort()
file_list.sort(key=lambda x: extract_last_number(os.path.basename(x)))

with open(output_txt, 'w', encoding='utf-8') as f:
    for filepath in file_list:
        f.write(filepath + '\n')

