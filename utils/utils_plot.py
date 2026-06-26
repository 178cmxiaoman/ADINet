import matplotlib.pyplot as plt


def plot_space_matrix(space, t):
    """
    用于绘制论文中Algorithm 4 ： Novel Nonlinear Dynamic Feature Extraction Based on Phase Space的相空间矩阵

    参数:
    space (np.array): 20x20 的相空间矩阵
    t (int): 当前时间延迟
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(space, cmap="hot", interpolation="nearest", vmax=20)
    plt.colorbar(label="Counts")
    plt.title(f"Phase Space Matrix (Time Delay = {t})")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.xticks(range(0, 20, 2))
    plt.yticks(range(0, 20, 2))
    plt.show()
