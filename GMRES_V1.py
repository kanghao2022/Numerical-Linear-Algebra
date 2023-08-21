'''
使用Python手动实现的GMRES算法，version 1
未设置重启
最小二乘问题使用Givens旋转进行求解
'''


# solution=array([ 2., -2.,  9.])}
# A = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
# b = np.array([2, 4, -1])


import numpy as np
from scipy.sparse import csr_matrix
import scipy.io as sio
import time


def dot(x, y):
    return np.dot(x, y)


def norm(x):
    return np.sqrt(np.inner(x, x))


def GMRES(A, b, eps, itermax, x0=None):

    # 初始化参数
    dim = b.size
    if x0 == None:
        x0 = np.zeros((dim,))

    v = np.zeros((dim, itermax + 1))
    h = np.zeros((itermax + 1, itermax))

    ksi = np.zeros((itermax+1,))
    e = np.zeros((dim, 1))
    e[0] = 1
    r0 = b - np.matmul(A, x0)
    beta = norm(r0)
    if (beta / norm(b)) < eps:
        res = x0
    v[0:, 0] = (1 / beta) * r0
    ksi[0] = beta * e[0]
    c = np.zeros(itermax+1, )
    s = np.zeros(itermax+1, )

    iter = 0

    R = np.zeros((itermax+1,itermax))

    for j in range(1, itermax + 1):
        iter += 1
        j -= 1                  #下标对齐
        w = np.dot(A, v[:, j])
        # v[:, j+1] = w           #用矩阵v的j+1列保存Av_j的结算结果

        for i in range(j+1):
            h[i, j] = np.dot(w, v[:, i])
            w = w - np.dot(h[i, j], v[:, i])

        h[j + 1, j] = np.linalg.norm(w)

        if h[j + 1, j] == 0:
            m = j
            break
        v[:, j + 1] = w / h[j + 1, j]

        R[:,j] = h[:,j]

        # givens旋转，得到上三角矩阵R
        for i in range(j):

            temp = c[i] * R[i, j] + s[i] * R[i + 1, j]
            R[i + 1, j] = (-s[i]) * R[i, j] + c[i] * R[i + 1, j]
            R[i, j] = temp


        # if h[j + 1, j - 1] == 0:
        #     c[j - 1] = 1
        #     s[j - 1] = 0

        if np.abs(R[j, j]) > np.abs(R[j+1, j]):
            tau = R[j + 1, j] / R[j, j]
            c[j] = 1 / np.sqrt(1 + tau * tau)
            s[j] = c[j] * tau

        else:
            tau = R[j, j] / R[j + 1, j]
            s[j] = 1 / np.sqrt(1 + tau * tau)
            c[j] = s[j] * tau


        R[j, j] = c[j] * R[j, j] + s[j] * R[j + 1, j]


        R[j + 1, j] = 0

        temp = c[j] * ksi[j]
        ksi[j+1] = -s[j] * ksi[j]
        ksi[j] = temp

        relres = np.abs(ksi[j+1] / beta)
        if relres < eps:
            m = j + 1
            break
    m = j + 1

    # 回代求解
    Z = np.dot(np.linalg.inv(R[:m,:m]), ksi[0:m])

    x = x0 + np.dot(v[:, 0:m], Z)

    # print(v[0:,0:m],"v")
    # print(h[:m,:m],'h')
    # print(R[:m,:m],'R')
    # print(ksi,"ksi")


    residual = norm(b - A@x)
    print(iter)

    return relres

# with open('TestArray_1e4.txt', 'r') as f:
#     # open为打开文件，r为读取
#     f = open('TestArray_1e4.txt', 'r')
#     # 逐行读取文件内容
#     lines = f.readlines()
#     rightHandsideArray = []
#     temp_strLst = lines[0].split()

#     for i in range(2, len(temp_strLst)):
#         rightHandsideArray.append(eval(temp_strLst[i]))
#     f.close()
#     b = np.array(rightHandsideArray)


if __name__ == '__main__':
    # solution=array([ 2., -2.,  9.])}
    # A = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
    # A = np.array([[1, 2, 3], [2, 5, 1], [4, 2, 1]])
    A = np.array([[1, 3, 2], [2, 5, 5], [3, 7, 1]], dtype=float)
    # m = A.shape[0]
    b = np.array([2, 4, -1])
    # A = sio.mmread('FDM_2D_Poisson_1e4.mtx').tocsr()
    # A = A.toarray()
    # b = sio.mmread('FDM_2D_Poisson_25_rhs1.mtx')
    # x0 = np.zeros((b.shape[0],1))
    # x0[0] = 1
    eps = 1e-10
    itermax = 10
    t1 = time.time()
    print(GMRES(A = A, b = b, eps = eps, itermax = itermax))
    t2 = time.time()
    print(t2 - t1)
