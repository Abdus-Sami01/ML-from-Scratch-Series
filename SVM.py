'''
SUPPORT VECTOR MACHINE — FROM SCRATCH
Pure Python. No numpy, no scipy, no sklearn.

Two complete implementations:
    1. LinearSVM  — primal form, SGD on hinge loss (fast, large data)
    2. KernelSVM  — dual form, SMO solver, supports Linear/Poly/RBF kernels

Math path:
    Primal QP → Lagrangian → KKT conditions → Dual QP
    → replace xᵢ·xⱼ with K(xᵢ,xⱼ)  (Mercer's theorem)
    → solve with SMO  (Sequential Minimal Optimization)
    → predict: ŷ = sign(Σᵢ αᵢ yᵢ K(xᵢ,x) + b)
'''


def make_matrix(rows, cols, fill=0.0):
    return [[fill] * cols for _ in range(rows)]

def shape(M):
    return (len(M), len(M[0]) if M else 0)

def transpose(M):
    r, c = shape(M)
    T = make_matrix(c, r)
    for i in range(r):
        for j in range(c):
            T[j][i] = M[i][j]
    return T

def mat_mul(A, B):
    r, k = shape(A)
    k2, c = shape(B)
    assert k == k2
    R = make_matrix(r, c)
    for i in range(r):
        for j in range(c):
            s = 0.0
            for m in range(k):
                s += A[i][m] * B[m][j]
            R[i][j] = s
    return R

def dot(a, b):
    '''Dot product of two flat lists (vectors).'''
    return sum(a[i] * b[i] for i in range(len(a)))

def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def vec_scale(a, s):
    return [a[i] * s for i in range(len(a))]

def norm_sq(a):
    '''||a||² — squared L2 norm.'''
    return sum(x * x for x in a)

def norm(a):
    return norm_sq(a) ** 0.5

def clip(x, lo, hi):
    return max(lo, min(hi, x))


E = 2.718281828459045

def exp(x):
    x = clip(x, -500, 500)
    return E ** x

def log(x):
    if x <= 0:
        return float('-inf')
    n = 0
    v = x
    while v >= E:
        v /= E; n += 1
    while v < 1.0 / E:
        v *= E; n -= 1
    t = (v - 1.0) / (v + 1.0)
    t2 = t * t
    s, term = 0.0, t
    for k in range(1, 200, 2):
        s += term / k
        term *= t2
        if abs(term / k) < 1e-15:
            break
    return n + 2.0 * s

def sign(x):
    '''Returns +1 or -1. Zero goes to +1 by convention.'''
    return 1 if x >= 0 else -1


class LCG:
    def __init__(self, seed=42):
        self.s = seed
        self.m = 2 ** 32
        self.a = 1664525
        self.c = 1013904223

    def rand_int(self):
        self.s = (self.a * self.s + self.c) % self.m
        return self.s

    def rand_float(self):
        return self.rand_int() / self.m

    def rand_normal(self):
        u1 = max(self.rand_float(), 1e-10)
        u2 = self.rand_float()
        pi = 3.141592653589793
        angle = 2 * pi * u2
        cos_val, term = 0.0, 1.0
        for k in range(1, 20):
            cos_val += term
            term *= -angle * angle / ((2*k) * (2*k-1))
        return ((-2.0 * log(u1)) ** 0.5) * cos_val

rng = LCG(seed=42)

def shuffle_indices(n):
    idx = list(range(n))
    for i in range(n-1, 0, -1):
        j = rng.rand_int() % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def kernel_linear(xi, xj):
    return dot(xi, xj)

def kernel_poly(xi, xj, degree=3, coef=1.0):
    return (dot(xi, xj) + coef) ** degree

def kernel_rbf(xi, xj, gamma=0.5):
    diff = vec_sub(xi, xj)
    return exp(-gamma * norm_sq(diff))

def build_kernel_matrix(X, kernel_fn):
    n = len(X)
    K = make_matrix(n, n)
    for i in range(n):
        for j in range(i, n):         
            val = kernel_fn(X[i], X[j])
            K[i][j] = val
            K[j][i] = val
    return K



def smo_solver(X, Y, K, C=1.0, tol=1e-4, max_passes=200):
    n = len(X)
    alphas = [0.0] * n     
    b = 0.0
    passes = 0

    def f(i):
        total = 0.0
        for j in range(n):
            if alphas[j] > 0:     
                total += alphas[j] * Y[j] * K[j][i]
        return total + b

    while passes < max_passes:
        num_changed = 0

        for i in range(n):
            Ei = f(i) - Y[i]


            kkt_violated = (Y[i] * Ei < -tol and alphas[i] < C) or \
                           (Y[i] * Ei >  tol and alphas[i] > 0)

            if not kkt_violated:
                continue

            j = i
            max_diff = 0.0
            for k in range(n):
                if k == i:
                    continue
                Ek = f(k) - Y[k]
                if abs(Ei - Ek) > max_diff:
                    max_diff = abs(Ei - Ek)
                    j = k
            if j == i:                  
                j = (i + 1) % n

            Ej = f(j) - Y[j]

            alpha_i_old = alphas[i]
            alpha_j_old = alphas[j]


            if Y[i] == Y[j]:
                L = max(0.0, alpha_i_old + alpha_j_old - C)
                H = min(C,   alpha_i_old + alpha_j_old)
            else:
                L = max(0.0, alpha_j_old - alpha_i_old)
                H = min(C,   C + alpha_j_old - alpha_i_old)

            if abs(H - L) < 1e-12:
                continue        


            eta = 2.0 * K[i][j] - K[i][i] - K[j][j]

            if eta >= 0:
                continue       

            alphas[j] = alpha_j_old - Y[j] * (Ei - Ej) / eta
            alphas[j] = clip(alphas[j], L, H)      

            if abs(alphas[j] - alpha_j_old) < 1e-8:
                continue      


            alphas[i] = alpha_i_old + Y[i] * Y[j] * (alpha_j_old - alphas[j])


            b1 = b - Ei \
                   - Y[i] * (alphas[i] - alpha_i_old) * K[i][i] \
                   - Y[j] * (alphas[j] - alpha_j_old) * K[i][j]

            b2 = b - Ej \
                   - Y[i] * (alphas[i] - alpha_i_old) * K[i][j] \
                   - Y[j] * (alphas[j] - alpha_j_old) * K[j][j]

            if 0 < alphas[i] < C:
                b = b1
            elif 0 < alphas[j] < C:
                b = b2
            else:
                b = (b1 + b2) / 2.0

            num_changed += 1

        if num_changed == 0:
            passes += 1
        else:
            passes = 0    

    return alphas, b



class KernelSVM:
    KERNELS = {
        'linear' : lambda gamma, degree, coef: (lambda xi, xj: kernel_linear(xi, xj)),
        'poly'   : lambda gamma, degree, coef: (lambda xi, xj: kernel_poly(xi, xj, degree, coef)),
        'rbf'    : lambda gamma, degree, coef: (lambda xi, xj: kernel_rbf(xi, xj, gamma)),
    }

    def __init__(self, C=1.0, kernel='rbf', gamma=0.5, degree=3,
                 coef=1.0, tol=1e-4, max_passes=200):
        self.C           = C
        self.kernel_name = kernel
        self.gamma       = gamma
        self.degree      = degree
        self.coef        = coef
        self.tol         = tol
        self.max_passes  = max_passes
        self.kernel_fn   = self.KERNELS[kernel](gamma, degree, coef)

        self.support_vectors  = []    
        self.support_labels   = []    
        self.support_alphas   = []    
        self.b                = 0.0
        self.n_support        = 0

    def fit(self, X, Y):
        print(f"  Building {len(X)}×{len(X)} Gram matrix ({self.kernel_name} kernel)...")
        K = build_kernel_matrix(X, self.kernel_fn)

        print(f"  Running SMO solver (C={self.C}, tol={self.tol})...")
        alphas, self.b = smo_solver(X, Y, K, self.C, self.tol, self.max_passes)

        sv_threshold = 1e-6
        for i in range(len(X)):
            if alphas[i] > sv_threshold:
                self.support_vectors.append(X[i])
                self.support_labels.append(Y[i])
                self.support_alphas.append(alphas[i])

        self.n_support = len(self.support_vectors)
        print(f"  Done. {self.n_support} support vectors found out of {len(X)} samples.")
        return self

    def decision_function(self, X):
        results = []
        for x in X:
            val = self.b
            for i in range(self.n_support):
                val += (self.support_alphas[i]
                        * self.support_labels[i]
                        * self.kernel_fn(self.support_vectors[i], x))
            results.append(val)
        return results

    def predict(self, X):
        '''ŷ = sign(f(x)) → +1 or -1'''
        return [sign(v) for v in self.decision_function(X)]

    def accuracy(self, X, Y):
        preds = self.predict(X)
        correct = sum(1 for p, y in zip(preds, Y) if p == y)
        return correct / len(Y)

    def score_report(self, X, Y, name=""):
        preds = self.predict(X)
        acc = sum(1 for p, y in zip(preds, Y) if p == y) / len(Y)
        tp = sum(1 for p, y in zip(preds, Y) if p ==  1 and y ==  1)
        tn = sum(1 for p, y in zip(preds, Y) if p == -1 and y == -1)
        fp = sum(1 for p, y in zip(preds, Y) if p ==  1 and y == -1)
        fn = sum(1 for p, y in zip(preds, Y) if p == -1 and y ==  1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        tag = f" [{name}]" if name else ""
        print(f"  Accuracy{tag}  : {acc*100:.2f}%")
        print(f"  Precision      : {precision:.4f}")
        print(f"  Recall         : {recall:.4f}")
        print(f"  F1 Score       : {f1:.4f}")
        print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")


class LinearSVM:
    def __init__(self, C=1.0, learning_rate=0.01, n_epochs=1000,
                 batch_size=32, tol=1e-6):
        self.C  = C
        self.lr = learning_rate
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.tol        = tol
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def _hinge_loss(self, X, Y):
        reg  = 0.5 * norm_sq(self.w)
        hinge = 0.0
        for xi, yi in zip(X, Y):
            margin = yi * (dot(self.w, xi) + self.b)
            hinge += max(0.0, 1.0 - margin)
        return reg + self.C * hinge

    def fit(self, X, Y):
        n_features = len(X[0])
        n = len(X)
        self.w = [rng.rand_normal() * 0.01 for _ in range(n_features)]
        self.b = 0.0

        prev_loss = float('inf')

        for epoch in range(self.n_epochs):
            idx = shuffle_indices(n)

            for start in range(0, n, self.batch_size):
                batch = idx[start: start + self.batch_size]
                dw = [0.0] * n_features
                db = 0.0
                m  = len(batch)

                for i in batch:
                    xi, yi = X[i], Y[i]
                    margin = yi * (dot(self.w, xi) + self.b)

                    if margin >= 1.0:
                        for k in range(n_features):
                            dw[k] += self.w[k]
                    else:
                        for k in range(n_features):
                            dw[k] += self.w[k] - self.C * yi * xi[k]
                        db += -self.C * yi

                for k in range(n_features):
                    self.w[k] -= self.lr * dw[k] / m
                self.b -= self.lr * db / m

            loss = self._hinge_loss(X, Y)
            self.loss_history.append(loss)

            if (epoch + 1) % 100 == 0 or epoch == self.n_epochs - 1:
                print(f"  Epoch {epoch+1:>5}/{self.n_epochs}  |  Loss: {loss:.6f}")

            if abs(prev_loss - loss) < self.tol:
                print(f"  Converged at epoch {epoch+1}.")
                break
            prev_loss = loss

        return self

    def decision_function(self, X):
        '''w·x + b for each sample.'''
        return [dot(self.w, x) + self.b for x in X]

    def predict(self, X):
        return [sign(v) for v in self.decision_function(X)]

    def accuracy(self, X, Y):
        preds = self.predict(X)
        return sum(1 for p, y in zip(preds, Y) if p == y) / len(Y)

    def score_report(self, X, Y, name=""):
        preds = self.predict(X)
        acc = sum(1 for p, y in zip(preds, Y) if p == y) / len(Y)
        tp = sum(1 for p, y in zip(preds, Y) if p ==  1 and y ==  1)
        tn = sum(1 for p, y in zip(preds, Y) if p == -1 and y == -1)
        fp = sum(1 for p, y in zip(preds, Y) if p ==  1 and y == -1)
        fn = sum(1 for p, y in zip(preds, Y) if p == -1 and y ==  1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        tag = f" [{name}]" if name else ""
        print(f"  Accuracy{tag}  : {acc*100:.2f}%")
        print(f"  Precision      : {precision:.4f}")
        print(f"  Recall         : {recall:.4f}")
        print(f"  F1 Score       : {f1:.4f}")
        print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")

    def weights_summary(self):
        print(f"  w = {[round(wi, 4) for wi in self.w]}")
        print(f"  b = {self.b:.4f}")
        print(f"  ||w|| = {norm(self.w):.4f}   (margin = {2/norm(self.w):.4f})")


def generate_linear_data(n=200, seed=0):
    gen = LCG(seed)
    X, Y = [], []
    for i in range(n):
        label = 1 if i % 2 == 0 else -1
        cx = 1.5 * label
        X.append([cx + gen.rand_normal() * 0.5,
                  cx + gen.rand_normal() * 0.5])
        Y.append(float(label))
    return X, Y

def generate_circular_data(n=200, seed=3):
    gen = LCG(seed)
    pi  = 3.141592653589793
    X, Y = [], []
    for i in range(n):
        angle = gen.rand_float() * 2 * pi
        if i % 2 == 0:
            r = 0.5 + gen.rand_normal() * 0.1   
            label = 1.0
        else:
            r = 1.5 + gen.rand_normal() * 0.1   
            label = -1.0
        X.append([r * cos_approx(angle), r * sin_approx(angle)])
        Y.append(label)
    return X, Y

def generate_xor_data(n=200, seed=7):
    gen = LCG(seed)
    X, Y = [], []
    centers = [(1,1), (-1,-1), (-1,1), (1,-1)]
    labels  = [1.0,    1.0,   -1.0,  -1.0]
    per = n // 4
    for c, lbl in zip(centers, labels):
        for _ in range(per):
            X.append([c[0] + gen.rand_normal() * 0.3,
                      c[1] + gen.rand_normal() * 0.3])
            Y.append(lbl)
    return X, Y

def cos_approx(x):
    x = x % (2 * 3.141592653589793)
    result, term = 1.0, 1.0
    for k in range(1, 15):
        term *= -x * x / ((2*k - 1) * (2*k))
        result += term
        if abs(term) < 1e-12:
            break
    return result

def sin_approx(x):
    '''sin(x) = cos(x - π/2)'''
    return cos_approx(x - 3.141592653589793 / 2)

def train_test_split(X, Y, test_ratio=0.2, seed=1):
    n = len(X)
    gen = LCG(seed)
    idx = list(range(n))
    for i in range(n-1, 0, -1):
        j = gen.rand_int() % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    cut = int(n * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return ([X[i] for i in tr], [X[i] for i in te],
            [Y[i] for i in tr], [Y[i] for i in te])


sep = "=" * 55

print(sep)
print("TEST 1 — LinearSVM (Primal SGD) | Linearly Separable Data")
print(sep)

X_lin, Y_lin = generate_linear_data(n=300, seed=0)
Xtr, Xte, Ytr, Yte = train_test_split(X_lin, Y_lin)

svm_linear = LinearSVM(C=1.0, learning_rate=0.01, n_epochs=500, batch_size=32)
svm_linear.fit(Xtr, Ytr)
print()
svm_linear.score_report(Xte, Yte, name="Linear SVM")
svm_linear.weights_summary()

print()
print(sep)
print("TEST 2 — KernelSVM (Linear Kernel) | Linearly Separable Data")
print(sep)

svm_k_lin = KernelSVM(C=1.0, kernel='linear', tol=1e-4, max_passes=100)
svm_k_lin.fit(Xtr, Ytr)
print()
svm_k_lin.score_report(Xte, Yte, name="Kernel Linear")

print()
print(sep)
print("TEST 3 — KernelSVM (RBF Kernel) | Concentric Circles")
print("  (Cannot be solved by any linear classifier)")
print(sep)

X_circ, Y_circ = generate_circular_data(n=200, seed=3)
Xtr2, Xte2, Ytr2, Yte2 = train_test_split(X_circ, Y_circ)

print("  [Control] LinearSVM on circular data (should fail):")
svm_fail = LinearSVM(C=1.0, learning_rate=0.01, n_epochs=300, batch_size=32)
svm_fail.fit(Xtr2, Ytr2)
acc_fail = svm_fail.accuracy(Xte2, Yte2)
print(f"  LinearSVM Accuracy: {acc_fail*100:.2f}%  ← should be near 50% (random)")

print()
print("  [RBF Kernel SVM]:")
svm_rbf = KernelSVM(C=5.0, kernel='rbf', gamma=1.5, tol=1e-4, max_passes=150)
svm_rbf.fit(Xtr2, Ytr2)
print()
svm_rbf.score_report(Xte2, Yte2, name="RBF Kernel")

print()
print(sep)
print("TEST 4 — KernelSVM (RBF Kernel) | XOR Pattern")
print("  (The classic non-linear benchmark)")
print(sep)

X_xor, Y_xor = generate_xor_data(n=200, seed=7)
Xtr3, Xte3, Ytr3, Yte3 = train_test_split(X_xor, Y_xor)

svm_xor = KernelSVM(C=2.0, kernel='rbf', gamma=1.0, tol=1e-4, max_passes=150)
svm_xor.fit(Xtr3, Ytr3)
print()
svm_xor.score_report(Xte3, Yte3, name="XOR + RBF")

print()
print(sep)
print("TEST 5 — KernelSVM (Polynomial Kernel, d=3) | XOR Pattern")
print(sep)

svm_poly = KernelSVM(C=2.0, kernel='poly', degree=3, coef=1.0, tol=1e-4, max_passes=150)
svm_poly.fit(Xtr3, Ytr3)
print()
svm_poly.score_report(Xte3, Yte3, name="Poly d=3")

print()
print(sep)
print("TEST 6 — Effect of C on RBF Kernel (Circles data)")
print("  Small C → wide margin, more violations allowed")
print("  Large C → narrow margin, penalise violations harder")
print(sep)

for C_val in [0.1, 1.0, 10.0, 100.0]:
    svm_c = KernelSVM(C=C_val, kernel='rbf', gamma=1.5,
                      tol=1e-4, max_passes=100)
    svm_c.fit(Xtr2, Ytr2)
    acc = svm_c.accuracy(Xte2, Yte2)
    print(f"  C={C_val:>6}  |  Accuracy: {acc*100:.2f}%"
          f"  |  Support vectors: {svm_c.n_support}")

print()
print(sep)
print("TEST 7 — Effect of γ on RBF Kernel (Circles data)")
print("  Large γ → local (complex boundary) → risk overfit")
print("  Small γ → global (smooth boundary)  → risk underfit")
print(sep)

for g_val in [0.1, 0.5, 1.5, 5.0, 20.0]:
    svm_g = KernelSVM(C=5.0, kernel='rbf', gamma=g_val,
                      tol=1e-4, max_passes=100)
    svm_g.fit(Xtr2, Ytr2)
    acc = svm_g.accuracy(Xte2, Yte2)
    print(f"  γ={g_val:>5}  |  Accuracy: {acc*100:.2f}%"
          f"  |  Support vectors: {svm_g.n_support}")
