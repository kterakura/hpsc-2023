import numpy as np
import matplotlib.pyplot as plt

# データファイルからデータを読み込む
def read_data(filename):
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    p = data[:, 2]
    u = data[:, 3]
    v = data[:, 4]
    return x, y, p, u, v

# 可視化
def visualize(x, y, p, u, v):
    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    P = p.reshape((len(np.unique(y)), len(np.unique(x))))
    U = u.reshape((len(np.unique(y)), len(np.unique(x))))
    V = v.reshape((len(np.unique(y)), len(np.unique(x))))

    plt.contourf(X, Y, P, alpha=0.5, cmap=plt.cm.coolwarm)
    plt.quiver(X[::2, ::2], Y[::2, ::2], U[::2, ::2], V[::2, ::2])
    plt.pause(.01)
    plt.clf()


# データファイルの読み込みと可視化
for n in range(500):
    filename = f'output/{n}.txt'
    x, y, p, u, v = read_data(filename)
    visualize(x, y, p, u, v)
plt.show()