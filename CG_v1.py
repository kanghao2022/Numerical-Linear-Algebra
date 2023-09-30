import numpy as np
from numpy.linalg import norm

def CG(A,b,x0,itermax,epsilon):
    n = b.shape[0]
    b = b.reshape(n,)

    r0 = b - np.dot(A,x0)
    p0 = r0

    
    iters=1
    while(iters<itermax):

        q = np.dot(A,p0)
        alpha = np.dot(r0,r0) / np.dot(q,p0)
        x0 += alpha * p0
        r1 = r0 - alpha * q
        beta = np.dot(r1,r1) / np.dot(r0,r0)
        p0 = r1 + np.dot(beta,p0)

        if(norm(r1) <= epsilon):
            break

        r0 = r1
        iters+=1
    return x0,iters


if __name__ == '__main__':

    matrixSize = 18
    def Atridiag(val_0, val_sup, val_inf, mSize):
        cen     = np.ones((1, mSize))*val_0
        sup     = np.ones((1, mSize-1))*val_sup
        inf     = np.ones((1, mSize-1))*val_inf
        diag_cen  = np.diagflat(cen, 0)
        diag_sup  = np.diagflat(sup, 1)
        diag_inf  = np.diagflat(inf, -1)
        return diag_cen + diag_sup + diag_inf

    A = Atridiag(2, -1, -1, matrixSize)
    A = np.array(A)

    b = np.ones((matrixSize, 1))
    guess = np.zeros((matrixSize,))
    x,n = CG(A,b,guess,10,1e-5)
    
    print(np.linalg.norm(A@x-b),n)