import torch

# 创建一个 4x4 的原始矩阵和一个 3x3 的子矩阵
original_matrix = torch.zeros(4, 4)
print(original_matrix)
submatrix = torch.ones(3, 3)
print(submatrix)

# 设置起始行和列
start_row = torch.tensor(1)
start_col = torch.tensor(1)

# 确保索引不超出范围
if (start_row + submatrix.size(0) <= original_matrix.size(0)) and (start_col + submatrix.size(1) <= original_matrix.size(1)):
    # 执行赋值操作
    print("start_row")
    print(start_row)
    print("start_row + submatrix.size(0)")
    print(start_row + submatrix.size(0))
    print("start_col")
    print(start_col)
    print("start_col+ submatrix.size(1)")
    print(start_col + submatrix.size(1))
    original_matrix[start_row:start_row + submatrix.size(0), start_col:start_col + submatrix.size(1)] = submatrix
    print(original_matrix)
else:
    print("索引超出范围，赋值操作无法完成。")
