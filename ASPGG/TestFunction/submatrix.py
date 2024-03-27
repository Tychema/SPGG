import torch
import random
def random_submatrix(matrix):
    #probability = random.uniform(0, 1)
    probability=0.45
    print("probability")
    print(probability)
    L_num= matrix.shape[0]
    # 根据概率值生成相应的数字
    if probability < 4 / (L_num * L_num):
        random_number = random.choice([0, 1, 2, 3])
        #左上角
        if  random_number==0:
            submatrix = matrix[0:2, 0:2].clone()
            return submatrix, 0, 0,0
        #右上角
        elif random_number==1:
            submatrix = matrix[0:2, L_num - 2:L_num].clone()
            return submatrix, 0, L_num - 2,0
        #左下角
        elif random_number==2:
            submatrix = matrix[L_num - 2:L_num, 0:2].clone()
            return submatrix, L_num - 2, 0,0
        #右下角
        else:
            submatrix = matrix[L_num - 2:L_num, L_num - 2:L_num].clone()
            return submatrix, L_num - 2, L_num - 2,0
    elif probability < (4 * (L_num - 2)) / (L_num * L_num) and probability >= 4 / (L_num * L_num):
        random_number = random.choice([0, 1, 2, 3])
        #上
        if  random_number==0:
            random_col = random.randint(1, L_num - 1)
            submatrix = matrix[0:2, random_col-1:random_col+2].clone()
            return submatrix, 0, random_col-1,1
        #左
        elif random_number==1:
            random_row = random.randint(1, L_num - 2)
            submatrix = matrix[random_row-1:random_row+2, 0:2].clone()
            return submatrix, 0, random_row-1,1
        #下
        elif random_number==2:
            random_col = random.randint(1, L_num - 1)
            submatrix = matrix[L_num-2:L_num,random_col-1:random_col+2].clone()
            return submatrix, L_num-2, random_col-1,1
        #右
        else:
            random_row = random.randint(1, L_num - 1)
            submatrix = matrix[random_row-1:random_row+2, L_num-2:L_num].clone()
            return submatrix, random_row-1, L_num-2,1
    else:
        start_row = torch.randint(0, L_num - 3 + 1, (1,))[0]
        start_col = torch.randint(0, L_num - 3 + 1, (1,))[0]
        submatrix = matrix[start_row:start_row + 3, start_col:start_col +3].clone()
        return submatrix, start_row, start_col,2

def process_submatrix(submatrix):
    # 这里可以进行一系列复杂的运算
    processed_submatrix = submatrix +1  # 示例运算，可以替换成实际的复杂运算
    return processed_submatrix

def insert_submatrix(original_matrix, submatrix, start_row, start_col):
    import torch.nn.functional as F
    original_matrix = F.pad(original_matrix, (1, 1, 1, 1), value=-1)
    if (start_row + submatrix.size(0) <= original_matrix.size(0)) and (
            start_col + submatrix.size(1) <= original_matrix.size(1)):
        original_matrix[start_row:start_row + submatrix.size(0), start_col:start_col + submatrix.size(1)] = submatrix
    else:
        print("error")
    return original_matrix[1:-1, 1:-1]

def random_submatrix2(matrix):
    import torch.nn.functional as F
    # 在周围扩展一圈，使用4进行填充
    L_num= matrix.shape[0]
    padded_matrix = F.pad(matrix, (1, 1, 1, 1), value=-1)
    start_row = torch.randint(0, L_num - 3 + 1 + 1, (1,))[0]
    start_col = torch.randint(0, L_num - 3 + 1 + 1, (1,))[0]
    submatrix = padded_matrix[start_row:start_row + 3, start_col:start_col + 3].clone()
    return submatrix, start_row, start_col


# 示例
L = 4
matrix = torch.rand((L, L))
print("Original Matrix:")
print(matrix)

submatrix, start_row, start_col = random_submatrix2(matrix)
print("Submatrix:")
print(submatrix)

#processed_submatrix = process_submatrix(submatrix)
process_matrix = process_submatrix(submatrix)
print("Processed Submatrix:")
print(process_matrix)

# 将处理后的子矩阵放回原来的位置
new_matrix=insert_submatrix(matrix,process_matrix, start_row, start_col)


print("new_matrix:")
print(new_matrix)
