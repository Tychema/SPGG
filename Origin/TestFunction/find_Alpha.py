import numpy as np
import matplotlib.pyplot as plt
import math
def first_function(x, y, alpha):
    if x > y:
        return (np.exp(alpha*y) / (2 * np.exp(alpha*x)))
    else:
        return ((2 * np.exp(alpha*y) - np.exp(alpha*x)) / (2 * np.exp(alpha*y)))

def second_function(x, y):
    return 1 / (1 + np.exp((x-y) / 0.5))


def find_alpha():
    # 生成测试数据
    x_values = np.linspace(-1, 20, 400)
    y_values = np.linspace(-1, 20, 400)
    alpha_values = np.linspace(1,2, 1000)
    #alpha_values = [1]

    # 计算误差
    errors = []
    #errors2= []
    for alpha in alpha_values:
        error = 0
        for x in x_values:
            for y in y_values:
                p1 = first_function(x, y, alpha)
                p2 = second_function(x, y)
                error += abs(p1 - p2)
                #error2= (p1 - p2) ** 2
                #errors.append(error2)
        errors.append(error)

    # 最优的alpha
    opt_alpha = alpha_values[np.argmin(errors)]
    print(f"Optimal alpha: {opt_alpha}")
    print(f"Minimum error: {np.min(errors)}")

    # 绘制误差图
    plt.figure(figsize=(10, 5))
    plt.plot(alpha_values, errors, label='Error vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Total Squared Error')
    plt.title('Error of First Function Relative to Second Function Across Alphas')
    plt.legend()
    plt.show()

def find_alpha2():
    # 生成测试数据
    x_values = np.linspace(-1, 20, 400)
    y_values = np.linspace(-1, 20, 400)
    alpha_values = np.linspace(0, 10, 1000)
    #alpha_values = [1]

    # 计算误差
    errors = []
    #errors2= []
    for alpha in alpha_values:
        error = 0
        for x in x_values:
            for y in y_values:
                p1 = first_function(x, y, alpha)
                p2 = second_function(x, y)
                error += (p1 - p2) ** 2
                #error2= (p1 - p2) ** 2
                #errors.append(error2)
        errors.append(error)

    # 最优的alpha
    opt_alpha = alpha_values[np.argmin(errors)]
    print(f"Optimal alpha: {opt_alpha}")
    print(f"Minimum error: {np.min(errors)}")

    # 绘制误差图
    plt.figure(figsize=(10, 5))
    plt.plot(alpha_values, errors, label='Error vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Total Squared Error')
    plt.title('Error of First Function Relative to Second Function Across Alphas')
    plt.legend()
    plt.show()

def show_function(alpha=0.5):
    # 设置alpha值
    # 生成x和y的数据
    # 生成x和y的数据
    x = np.linspace(-1, 20, 200)
    y = np.linspace(-1, 20, 200)
    x, y = np.meshgrid(x, y)

    # 计算每个点的函数值
    z1 = np.zeros_like(x)
    z2 = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z1[i, j] = first_function(x[i, j], y[i, j], alpha)
            z2[i, j] = second_function(x[i, j], y[i, j])

    # 绘制三维图
    fig = plt.figure(figsize=(12, 6))

    # 绘制第一个函数
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(x, y, z1, cmap='viridis')
    ax.set_title('First Function')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 绘制第二个函数
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(x, y, z2, cmap='viridis')
    ax.set_title('Second Function')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()

def show_error(alpha):
    # 设置alpha值

    # 生成x和y的数据
    x = np.linspace(-1, 20, 200)
    y = np.linspace(-1, 20, 200)
    x, y = np.meshgrid(x, y)

    # 计算每个点的函数值
    p1 = np.zeros_like(x)
    p2 = np.zeros_like(x)
    error = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            p1[i, j] = first_function(x[i, j], y[i, j], alpha)
            p2[i, j] = second_function(x[i, j], y[i, j])
            error[i, j] = (p1[i, j] - p2[i, j])

    # 绘制误差的三维图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(x, y, error, cmap='viridis')
    ax.set_title('Error Surface: $(p_1 - p_2)^2$')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Error')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()
if __name__ == '__main__':
    find_alpha()
    #find_alpha2()
    #show_function(1)
    #show_error(2)