def Cholesky(self, ztol=1.0e-5):
    # Computes the upper triangular Cholesky factorization of
    # a positive definite matrix.
    res = matrix([[]])
    res.zero(self.dimx, self.dimx)
    for i in range(self.dimx):
        S = sum([(res.value[k][i]) ** 2 for k in range(i)])
        d = self.value[i][i] - S
        if abs(d) < ztol:
            res.value[i][i] = 0.0
        else:
            if d < 0.0:
                raise ValueError("Matrix not positive-definite")
            res.value[i][i] = sqrt(d)
        for j in range(i + 1, self.dimx):
            S = sum([res.value[k][i] * res.value[k][j] for k in range(self.dimx)])
            if abs(S) < ztol:
                S = 0.0
            try:
                res.value[i][j] = (self.value[i][j] - S) / res.value[i][i]
            except:
                raise ValueError("Zero diagonal")
    return res


def CholeskyInverse(self):
    # Computes inverse of matrix given its Cholesky upper Triangular
    # decomposition of matrix.
    res = matrix([[]])
    res.zero(self.dimx, self.dimx)
    # Backward step for inverse.
    for j in reversed(range(self.dimx)):
        tjj = self.value[j][j]
        S = sum([self.value[j][k] * res.value[j][k] for k in range(j + 1, self.dimx)])
        res.value[j][j] = 1.0 / tjj ** 2 - S / tjj
        for i in reversed(range(j)):
            res.value[j][i] = res.value[i][j] = -sum(
                [self.value[i][k] * res.value[k][j] for k in range(i + 1, self.dimx)]) / self.value[i][i]
    return res


def inverse(self):
    aux = self.Cholesky()
    res = aux.CholeskyInverse()
    return res


def __repr__(self):
    return repr(self.value)

)