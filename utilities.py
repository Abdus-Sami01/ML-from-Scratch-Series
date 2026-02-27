from math import log as _log, factorial as _factorial

E  = 2.718281828459045
PI = 3.141592653589793

def exp(x):
    return E ** x

def log(x):
    if x <= 0:
        raise ValueError(f"log undefined for x={x}")
    return _log(x)

def log2(x):
    return log(x) / log(2)

def sqrt(x):
    if x < 0:
        raise ValueError(f"sqrt undefined for x={x}")
    return x ** 0.5

def power(base, exp):
    return base ** exp

def absolute(x):
    return x if x >= 0 else -x

def factorial(n):
    return _factorial(n)

def sign(x):
    return 1 if x >= 0 else -1


def clip(x, low, high):
    return min(high, max(x, low))

def floor(x):
    return int(x) if x >= 0 else int(x) - (1 if x != int(x) else 0)

def ceil(x):
    return int(x) if x == int(x) else floor(x) + 1

def round_to(x, decimals=4):
    factor = 10 ** decimals
    return floor(x * factor + 0.5) / factor


def cos(x, terms=15):
    result = 0
    for n in range(terms):
        result += ((-1) ** n) * (x ** (2 * n)) / factorial(2 * n)
    return result

def sin(x, terms=15):
    result = 0
    for n in range(terms):
        result += ((-1) ** n) * (x ** (2 * n + 1)) / factorial(2 * n + 1)
    return result

def mean(x):
    if len(x) == 0:
        raise ValueError("mean of empty list")
    return sum(x) / len(x)

def variance(x, ddof=0):
    if len(x) == 0:
        return 1e-9
    m = mean(x)
    v = sum((xi - m) ** 2 for xi in x) / (len(x) - ddof)
    return v if v != 0 else 1e-9

def std(x, ddof=0):
    return sqrt(variance(x, ddof))

def median(x):
    s = sorted(x)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2

def mode(x):
    counts = {}
    for v in x:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])

def covariance(x, y):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)

def correlation(x, y):
    denom = std(x) * std(y)
    if denom == 0:
        return 0.0
    return covariance(x, y) / denom


def gaussian_pdf(x, mu, var):
    coeff    = 1 / sqrt(2 * PI * var)
    exponent = -((x - mu) ** 2) / (2 * var)
    return coeff * exp(exponent)

def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    result = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            result -= p * log2(p)
    return result

def gini_impurity(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    return 1.0 - sum((c / total) ** 2 for c in counts)


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("dot: length mismatch")
    return sum(ai * bi for ai, bi in zip(a, b))

def vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def vec_sub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]

def scalar_mul(s, v):
    return [s * vi for vi in v]

def norm(v, p=2):
    return sum(absolute(vi) ** p for vi in v) ** (1 / p)

def mat_mul(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError("mat_mul: incompatible shapes")
    C = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return C

def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def col(matrix, j):
    return [row[j] for row in matrix]

def row_means(matrix):
    return [mean(row) for row in matrix]

def col_means(matrix):
    return [mean(col(matrix, j)) for j in range(len(matrix[0]))]

def matrix_mean(matrix):
    flat = [v for row in matrix for v in row]
    return mean(flat)


class LCG:
    def __init__(self, seed=42):
        self.seed = seed
        self.a = 1103515245
        self.c = 12345
        self.m = 2 ** 31

    def _advance(self):
        self.seed = (self.a * self.seed + self.c) % self.m
        return self.seed

    def next_int(self, low=0, high=100):
        return low + (self._advance() % (high - low + 1))

    def next_float(self, low=0.0, high=1.0):
        return low + (self._advance() / self.m) * (high - low)

    def next_range(self, lo, hi):
        return lo + self.next_float() * (hi - lo)

    def next_gaussian(self):
        """Box-Muller transform. u1, u2 must be in (0,1)."""
        u1 = self.next_float(0, 1)
        u2 = self.next_float(0, 1)
        u1 = max(u1, 1e-9)
        return sqrt(-2.0 * log(u1)) * cos(2 * PI * u2)

    def shuffle(self, lst):
        for i in range(len(lst) - 1, 0, -1):
            j = self.next_int(0, i)
            lst[i], lst[j] = lst[j], lst[i]
        return lst

    def shuffle_indices(self, n):
        return self.shuffle(list(range(n)))


def train_test_split(X, Y, test_ratio=0.2, seed=42):
    rng     = LCG(seed)
    n       = len(X)
    indices = rng.shuffle_indices(n)

    test_size    = int(n * test_ratio)
    test_idx     = indices[:test_size]
    train_idx    = indices[test_size:]

    return (
        [X[i] for i in train_idx],
        [X[i] for i in test_idx],
        [Y[i] for i in train_idx],
        [Y[i] for i in test_idx],
    )

def unique(lst):
    seen, out = set(), []
    for v in lst:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def count_values(lst):
    counts = {}
    for v in lst:
        counts[v] = counts.get(v, 0) + 1
    return counts

def argmax(lst):
    best_i, best_v = 0, lst[0]
    for i, v in enumerate(lst):
        if v > best_v:
            best_v, best_i = v, i
    return best_i

def flatten(matrix):
    return [v for row in matrix for v in row]

def zip_cols(matrix):
    for j in range(len(matrix[0])):
        yield [matrix[i][j] for i in range(len(matrix))]


def accuracy(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

def per_class_accuracy(y_true, y_pred):
    classes = sorted(unique(y_true))
    for c in classes:
        total   = sum(1 for y in y_true if y == c)
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        print(f"  class {c}: {correct}/{total} = {correct/total:.2%}")

def confusion_matrix(y_true, y_pred):
    classes = sorted(unique(y_true))
    idx     = {c: i for i, c in enumerate(classes)}
    n       = len(classes)
    cm      = [[0] * n for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        cm[idx[yt]][idx[yp]] += 1
    return cm, classes

def mean_squared_error(y_true, y_pred):
    return mean([(a - b) ** 2 for a, b in zip(y_true, y_pred)])

def mean_absolute_error(y_true, y_pred):
    return mean([absolute(a - b) for a, b in zip(y_true, y_pred)])