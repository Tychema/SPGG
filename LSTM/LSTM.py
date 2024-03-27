import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 示例用法
#L_num = 100
L_num = 100
batch_size = 1
num_channels = 3
width = 100
height = 100
input_size = (batch_size, num_channels, width, height)
hidden_size = 50
num_classes = 1
num_channels = 1
#hidden_size = 1000
output_size = L_num*L_num


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, cnn_output, profit_matrix):
        # Apply linear transformations
        transformed_output = torch.tanh(self.linear1(cnn_output))
        attn_weights = F.softmax(self.linear2(transformed_output), dim=1)

        weighted_output = torch.bmm(attn_weights.permute(0, 2, 1), profit_matrix)
        return weighted_output.squeeze(1)


class CNNModel(nn.Module):
    def __init__(self, input_shape, num_channels, num_classes):
        super(CNNModel, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Attention mechanism
        self.attention = Attention(64)  # Set hidden_size to the appropriate value

        # Fully connected layer for binary classification
        self.fc = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, profit_matrix):
        # Move data to CUDA
        x = x.to(device)
        profit_matrix = profit_matrix.to(device)

        # CNN forward pass
        x = self.cnn(x)
        print(x.shape)
        # Reshape for Attention
        x = x.view(x.size(0), 64, x.size(2) * x.size(3))  # Flatten spatial dimensions
        print(x.shape)
        # Apply attention mechanism to CNN output and profit matrix
        weighted_output = self.attention(x, profit_matrix)

        # Fully connected layer with sigmoid activation for binary classification
        x = self.fc(weighted_output)
        x = self.sigmoid(x)

        return x





# Create model instance
model = CNNModel(input_size, num_channels, num_classes).to(device)







# 生成虚拟数据，实际应用中需要替换为真实数据
def generate_data(L_num):
    cooperation_matrix = torch.randint(2, size=(1,1,L_num, L_num)).float()
    payoff_matrix = torch.rand(1,1,L_num, L_num)
    return cooperation_matrix, payoff_matrix

# 构建LSTM模型


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),  lr=1)

# 模拟循环迭代
max_iterations = 1

# 生成初始状态
t_matrix, payoff_matrix = generate_data(L_num)

# 循环迭代
for iteration in range(max_iterations):
    print(t_matrix.shape)
    # 预测下一个状态
    predicted_state = model(t_matrix,payoff_matrix)

    # 获取真实的下一个状态，这里使用随机生成的数据作为示例
    t_matrix, payoff_matrix = generate_data(L_num)


    print("predicted_state:")
    print(predicted_state)
    print("t_matrix:")
    print(t_matrix)
    loss = criterion(predicted_state, t_matrix)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for name, param in model.named_parameters():
        print(f"Parameter: {name}")
        print(f"Requires Grad: {param.requires_grad}")
        print(f"Gradient: {param.grad}")
        print("=" * 30)
    #
    # # 更新当前状态
    # current_state = next_state