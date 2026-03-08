from utilities import (
    mean, std, variance, sqrt, absolute,
    dot, norm, transpose, mat_mul, mat_vec_mul,
    LCG, train_test_split, gen_blobs,
)

''' Pca is something bit different requiring Linear Algebra more than anything else,
let's first figure out our linear algebra helper function'''

def vec_scale(v, s):
    return [vi * s for vi in v]

def vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def vec_sub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]

def outer(u, v):
    return [[ui * vj for vj in v] for ui in u]

def mat_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]

def mat_scale(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))]
            for i in range(len(A))]

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def covariance_matrix(X):
    '''
    C = (1/n) * X_centered^T * X_centered
    Shape: (n_features, n_features)
    C[i][j] = covariance between feature i and feature j
    Diagonal: variances of each feature
    '''
    n  = len(X)
    nf = len(X[0])
    means = [mean([X[i][j] for i in range(n)]) for j in range(nf)]
    Xc    = [[X[i][j] - means[j] for j in range(nf)] for i in range(n)]
    Xt    = transpose(Xc)
    C     = mat_mul(Xt, Xc)
    return [[C[i][j] / n for j in range(nf)] for i in range(nf)], means


def power_iteration(C, rng, n_iter=1000, tol=1e-10):
    '''
    Finds the eigenvector corresponding to the largest eigenvalue.

    Algorithm:
        b_0 = random unit vector
        b_{k+1} = C * b_k / ||C * b_k||

    After convergence:
        eigenvalue  λ = b^T * C * b   (Rayleigh quotient)
        eigenvector v = b
    '''
    nf = len(C)
    b  = [rng.next_gaussian() for _ in range(nf)]
    b_norm = norm(b, p=2)
    b  = [bi / b_norm for bi in b]

    for _ in range(n_iter):
        b_new      = mat_vec_mul(C, b)
        b_new_norm = norm(b_new, p=2)
        if b_new_norm < 1e-12:
            break
        b_new = [bi / b_new_norm for bi in b_new]
        shift = norm(vec_sub(b_new, b), p=2)
        b     = b_new
        if shift < tol:
            break

    Cb     = mat_vec_mul(C, b)
    lam    = dot(b, Cb)
    return lam, b

def deflate(C, lam, v):
    '''
    Remove the contribution of eigenvector v from C.
    C_new = C - λ * v * v^T
    After deflation, power iteration on C_new finds
    the next largest eigenvector.
    '''
    rank1 = mat_scale(outer(v, v), lam)
    return mat_sub(C, rank1)

def eigen_decomposition(C, n_components, seed=42):
    '''
    Find top-n_components eigenvectors via repeated
    power iteration + deflation.
    Returns (eigenvalues, eigenvectors) sorted descending.
    '''
    rng         = LCG(seed)
    C_remaining = [row[:] for row in C]
    eigenvalues  = []
    eigenvectors = []

    for _ in range(n_components):
        lam, v = power_iteration(C_remaining, rng)
        eigenvalues.append(lam)
        eigenvectors.append(v)
        C_remaining = deflate(C_remaining, lam, v)

    return eigenvalues, eigenvectors


class PCA:
    def __init__(self, n_components=2, seed=42):
        self.n_components  = n_components
        self.seed          = seed
        self.components_   = None  
        self.eigenvalues_  = None
        self.explained_variance_ratio_ = None
        self.mean_         = None

    def fit(self, X):
        C, self.mean_ = covariance_matrix(X)
        nf            = len(X[0])
        n_comp        = min(self.n_components, nf)

        lams, vecs = eigen_decomposition(C, n_comp, self.seed)

        total_var = sum(C[i][i] for i in range(nf))

        self.eigenvalues_ = lams
        self.components_  = vecs
        self.explained_variance_ratio_ = (
            [l / total_var for l in lams] if total_var > 0 else [0.0]*n_comp
        )
        return self

    def transform(self, X):
        '''
        Project X onto principal components.
        Z = X_centered * W
        where W columns are eigenvectors.
        '''
        Xc = [[X[i][j] - self.mean_[j] for j in range(len(X[0]))]
              for i in range(len(X))]
        return [[dot(x, pc) for pc in self.components_] for x in Xc]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        '''
        Reconstruct approximate X from projected Z.
        X_approx = Z * W^T + mean
        '''
        nf = len(self.mean_)
        return [[sum(Z[i][k] * self.components_[k][j]
                     for k in range(self.n_components))
                 + self.mean_[j]
                 for j in range(nf)]
                for i in range(len(Z))]

    def score_report(self, name=''):
        tag = f' [{name}]' if name else ''
        print(f'  PCA{tag}')
        total = 0.0
        for i, (lam, ratio) in enumerate(
                zip(self.eigenvalues_, self.explained_variance_ratio_)):
            total += ratio
            bar = '&' * int(ratio * 40)
            print(f'  PC{i+1}  eigenvalue={lam:>8.4f}  '
                  f'var={ratio*100:>5.1f}%  cumulative={total*100:>5.1f}%  {bar}')


def reconstruction_error(X, Z, pca):
    X_approx = pca.inverse_transform(Z)
    return mean([
        norm(vec_sub(X[i], X_approx[i]), p=2)**2
        for i in range(len(X))
    ])


def gen_correlated(n=300, seed=5):
    '''
    2D data with strong correlation between features.
    PCA should find one dominant direction.
    '''
    rng = LCG(seed); X = []
    for _ in range(n):
        x1 = rng.next_gaussian() * 2
        x2 = x1 * 1.5 + rng.next_gaussian() * 0.4
        X.append([x1, x2])
    return X

def gen_high_dim(n=200, n_features=10, n_informative=2, seed=7):
    '''
    High-dimensional data where only a few dimensions carry signal.
    Remaining dimensions are pure noise.
    '''
    rng = LCG(seed); X = []
    for _ in range(n):
        signal = [rng.next_gaussian() * 3 for _ in range(n_informative)]
        noise  = [rng.next_gaussian() * 0.2 for _ in range(n_features - n_informative)]
        X.append(signal + noise)
    return X

sep = '='*52

print(sep)
print('TEST 1 — PCA on correlated 2D data')
print('  Strong x1-x2 correlation → 1 dominant component')
print(sep)
X1 = gen_correlated(300)
pca1 = PCA(n_components=2)
Z1   = pca1.fit_transform(X1)
print()
pca1.score_report(name='correlated 2D')
err = reconstruction_error(X1, Z1, pca1)
print(f'  Reconstruction error : {err:.6f}  (should be ~0 with 2 PCs)')

print()
print(sep)
print('TEST 2 — Dimensionality reduction: 10D → 2D')
print('  Only 2 of 10 features carry signal, rest is noise')
print(sep)
X2 = gen_high_dim(200, n_features=10, n_informative=2)
pca2_full = PCA(n_components=10)
pca2_full.fit(X2)
print()
pca2_full.score_report(name='10 components')

print()
pca2 = PCA(n_components=2)
Z2   = pca2.fit_transform(X2)
err2 = reconstruction_error(X2, Z2, pca2)
total_var_kept = sum(pca2.explained_variance_ratio_)
print(f'  Keeping 2 of 10 components:')
print(f'    Variance retained  : {total_var_kept*100:.1f}%')
print(f'    Reconstruction MSE : {err2:.4f}')

print()
print(sep)
print('TEST 3 — PCA for visualisation: 3D blobs → 2D')
print(sep)
X3_raw, Y3 = gen_blobs(300, n_clusters=3, spread=0.6, seed=1)
X3 = [x + [x[0]*0.3 + x[1]*0.2] for x in X3_raw]
pca3 = PCA(n_components=2)
Z3   = pca3.fit_transform(X3)
print()
pca3.score_report(name='3D→2D')
print(f'\n  First 5 projected points (2D):')
for i in range(5):
    print(f'    [{Z3[i][0]:>7.3f}, {Z3[i][1]:>7.3f}]  true_label={Y3[i]}')

print()
print(sep)
print('TEST 4 — n_components vs reconstruction error')
print(sep)
X4 = gen_high_dim(200, n_features=8, n_informative=3)
print()
for nc in range(1, 9):
    pca_nc = PCA(n_components=nc)
    Z_nc   = pca_nc.fit_transform(X4)
    err_nc = reconstruction_error(X4, Z_nc, pca_nc)
    var    = sum(pca_nc.explained_variance_ratio_)
    bar    = '&' * int(var * 20)
    print(f'  n={nc}  var_kept={var*100:>5.1f}%  recon_err={err_nc:.4f}  {bar}')

print()
print(sep)
print('TEST 5 — PCA whitening check')
print('  After PCA, projected dimensions should be uncorrelated')
print(sep)
X5 = gen_correlated(500)
pca5 = PCA(n_components=2)
Z5   = pca5.fit_transform(X5)
z1   = [z[0] for z in Z5]
z2   = [z[1] for z in Z5]
m1, m2 = mean(z1), mean(z2)
cov_z1z2 = mean([(z1[i]-m1)*(z2[i]-m2) for i in range(len(z1))])
print()
print(f'  Var(PC1)          : {variance(z1):.4f}')
print(f'  Var(PC2)          : {variance(z2):.4f}')
print(f'  Cov(PC1, PC2)     : {cov_z1z2:.6f}  (should be ~0)')
print(f'  Original Cov(x1,x2): {mean([(X5[i][0]-mean([x[0] for x in X5]))*(X5[i][1]-mean([x[1] for x in X5])) for i in range(len(X5))]):.4f}')
