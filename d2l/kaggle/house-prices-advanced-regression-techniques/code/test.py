import pandas as pd
import numpy as np
import torch

# --- 1. 创建一个已经清洗过、全部为数值的示例 DataFrame ---
# 假设你的 DataFrame 已经经过了独热编码、标准化、空值填充等所有预处理步骤
data = {
    'feature_1': [1.2, 2.5, 3.1, 4.8, 5.0],
    'feature_2': [0.1, 0.5, 0.9, 1.2, 1.5],
    'feature_3': [100, 200, 150, 120, 180],
    'onehot_catA': [1, 0, 1, 0, 1],
    'onehot_catB': [0, 1, 0, 1, 0],
    'target': [0, 1, 0, 1, 0] # 假设这是一个二分类目标，或回归目标
}
df_cleaned = pd.DataFrame(data)

print("--- 已经清洗过的数值型 DataFrame ---")
print(df_cleaned)
print("\nDataFrame 列类型 (应该全部是数值类型):")
print(df_cleaned.dtypes)
print("\nDataFrame 空值计数 (应该全部为0):")
print(df_cleaned.isnull().sum())


# --- 2. 分离特征 (X) 和目标变量 (y) ---
# 假设 'target' 是你的目标变量
X_df = df_cleaned.drop('target', axis=1)
y_df = df_cleaned['target']

print("\n--- 特征 DataFrame (X_df) ---")
print(X_df)
print("\n--- 目标 Series (y_df) ---")
print(y_df)


# --- 3. 将 Pandas DataFrame/Series 转换为 NumPy 数组 ---
# 使用 .values 属性即可
X_numpy = X_df.values
y_numpy = y_df.values

print(f"\n--- 转换为 NumPy 数组后的形状 ---")
print(f"X_numpy 形状: {X_numpy.shape}")
print(f"y_numpy 形状: {y_numpy.shape}")
print("\nX_numpy (前5行):\n", X_numpy[:5])
print("\ny_numpy:\n", y_numpy)


# --- 4. 将 NumPy 数组转换为 PyTorch 张量 ---
# 关键步骤：使用 torch.tensor()
# 通常，特征数据使用 torch.float32
# 目标数据根据任务类型：
#   - 回归任务：torch.float32
#   - 分类任务：如果标签是整数ID，使用 torch.long (如0, 1, 2...)
#               如果标签是0/1浮点数（如二分类），使用 torch.float32

# 对于特征 X:
X_tensor = torch.tensor(X_numpy, dtype=torch.float32)

# 对于目标 y (假设是回归任务或二分类，转换为浮点数)：
y_tensor = torch.tensor(y_numpy, dtype=torch.float32)

# 如果是分类任务，且y是整数标签（如 0, 1, 2...），则dtype应该是torch.long
# y_tensor_classification = torch.tensor(y_numpy, dtype=torch.long)

# 对于某些 PyTorch 损失函数，目标张量需要是二维的 (N, 1) 而不是一维的 (N,)
# 可以使用 .view(-1, 1) 来重塑
y_tensor = y_tensor.view(-1, 1) # 或者 y_tensor.unsqueeze(1)

print(f"\n--- 最终转换为 PyTorch 张量后的形状 ---")
print(f"X_tensor 形状: {X_tensor.shape}")
print(f"y_tensor 形状: {y_tensor.shape}")

print("\nX_tensor (前5行):")
print(X_tensor[:5])
print("\ny_tensor:")
print(y_tensor)