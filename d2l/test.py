import torch
import os

# 创建数据目录
os.makedirs(os.path.join('..', 'data'), exist_ok=True)

# 写入CSV文件
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,175000\n')
    f.write('NA,NA,140000\n')

import pandas as pd
data = pd.read_csv(data_file)

# 分离输入和输出
input, output = data.iloc[:, 0:2], data.iloc[:, 2]

# 修正拼写错误：numeric_only（原错误：numberic_only）
input = input.fillna(input.mean(numeric_only=True))
print(input)