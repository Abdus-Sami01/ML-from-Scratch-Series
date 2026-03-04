from utilities import (
    mean, variance, unique, count_values, LCG,
    train_test_split, mean_squared_error, mean_absolute_error,
    accuracy, per_class_accuracy, sample_with_replacement, log2,
)

'''
the core idea is to build trees sequentially. Each tree fits the RESIDUALS
(errors) of all previous trees combined.
Regression:
F0(x)  = mean(Y)                        initial prediction
r1     = Y - F0(x)                      residuals
h1     = DecisionTree.fit(X, r1)        fit tree to residuals
F1(x)  = F0(x) + lr * h1(x)            update
r2     = Y - F1(x)                      new residuals
h2     = DecisionTree.fit(X, r2)        fit to new residuals
...
final-result will something like  FM(x)  = F0 + lr*h1 + lr*h2 + ... + lr*hM

classification (binary):
this uses log-odds so we will have to convert to probabilities using sigmoid first
Residuals = Y - P  where P = sigmoid(F(x))
Leaf values adjusted by 2nd order Taylor expansion of log-loss.
Multiclass:
One set of trees per class (one-vs-rest).
Residuals = one_hot(Y) - softmax(F(x))
'''

E = 2.718281828459045

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + E ** (-x))

def softmax(scores):
    m = max(scores)
    e = [E ** (s - m) for s in scores]
    t = sum(e)
    return [v / t for v in e]

def log(x):
    if x <= 0: return -1e15
    from utilities import log as _log
    return _log(x)


class RegressionTree:

    def __init__(self, max_depth=3, min_samples=2):
        self.max_depth   = max_depth
        self.min_samples = min_samples

    def fit(self, X, R):
        self.nf   = len(X[0])
        self.root = self._build(X, R, 0)
        return self

    def _mse(self, R):
        if not R: return 0.0
        m = mean(R)
        return mean([(r - m)**2 for r in R])

    def _best_split(self, X, R):
        best = (-1e15, None, None)
        n    = len(R)
        parent = self._mse(R)
        for f in range(self.nf):
            vals = sorted(set(x[f] for x in X))
            for i in range(len(vals) - 1):
                t  = (vals[i] + vals[i+1]) / 2
                li = [k for k in range(n) if X[k][f] <= t]
                ri = [k for k in range(n) if X[k][f] >  t]
                if len(li) < self.min_samples or len(ri) < self.min_samples:
                    continue
                Rl, Rr = [R[k] for k in li], [R[k] for k in ri]
                gain = parent - (len(Rl)/n * self._mse(Rl) + len(Rr)/n * self._mse(Rr))
                if gain > best[0]:
                    best = (gain, f, t)
        return best[1], best[2]

    def _build(self, X, R, d):
        if len(R) < self.min_samples or d >= self.max_depth:
            return ('leaf', mean(R))
        f, t = self._best_split(X, R)
        if f is None:
            return ('leaf', mean(R))
        n  = len(R)
        li = [i for i in range(n) if X[i][f] <= t]
        ri = [i for i in range(n) if X[i][f] >  t]
        return ('node', f, t,
                self._build([X[i] for i in li], [R[i] for i in li], d+1),
                self._build([X[i] for i in ri], [R[i] for i in ri], d+1))

    def _pred_one(self, x, node):
        if node[0] == 'leaf': return node[1]
        _, f, t, left, right = node
        return self._pred_one(x, left if x[f] <= t else right)

    def predict(self, X):
        return [self._pred_one(x, self.root) for x in X]


class GradientBoostingRegressor:
    '''
    Parameters
    ----------
    n_estimators : number of boosting rounds (trees)
    learning_rate: shrinkage scales each tree's contribution
                   small lr + more trees = better generalisation
    max_depth    : depth of each weak learner tree
    subsample    : fraction of training data used per tree
                   < 1.0 adds stochasticity (Stochastic GB)
    '''

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples=2, subsample=1.0, seed=42):
        self.n_estimators  = n_estimators
        self.lr            = learning_rate
        self.max_depth     = max_depth
        self.min_samples   = min_samples
        self.subsample     = subsample
        self.seed          = seed
        self.trees         = []
        self.F0            = None

    def fit(self, X, Y):
        Y = [float(y) for y in Y]
        n = len(X)
        rng = LCG(self.seed)

        self.F0 = mean(Y)
        F = [self.F0] * n         

        self.trees = []

        for m in range(self.n_estimators):
            R = [Y[i] - F[i] for i in range(n)]

            if self.subsample < 1.0:
                k   = max(1, int(n * self.subsample))
                idx = sample_with_replacement(n, k, rng)
                Xs  = [X[i] for i in idx]
                Rs  = [R[i] for i in idx]
            else:
                Xs, Rs = X, R

            tree = RegressionTree(self.max_depth, self.min_samples)
            tree.fit(Xs, Rs)
            self.trees.append(tree)

            update = tree.predict(X)
            F = [F[i] + self.lr * update[i] for i in range(n)]

        return self

    def predict(self, X):
        F = [self.F0] * len(X)
        for tree in self.trees:
            update = tree.predict(X)
            F = [F[i] + self.lr * update[i] for i in range(len(X))]
        return F

    def score_report(self, X, Y, name=''):
        Y    = [float(y) for y in Y]
        preds = self.predict(X)
        mse  = mean_squared_error(Y, preds)
        mae  = mean_absolute_error(Y, preds)
        ym   = mean(Y)
        ss_res = sum((a-b)**2 for a,b in zip(Y,preds))
        ss_tot = sum((y-ym)**2 for y in Y)
        r2   = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        tag  = f' [{name}]' if name else ''
        print(f'  MSE{tag}: {mse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}')


class GradientBoostingClassifier:
    
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples=2, subsample=1.0, seed=42):
        self.n_estimators = n_estimators
        self.lr           = learning_rate
        self.max_depth    = max_depth
        self.min_samples  = min_samples
        self.subsample    = subsample
        self.seed         = seed

    def fit(self, X, Y):
        self.classes = sorted(unique(Y))
        self.label_map = {c: i for i, c in enumerate(self.classes)}

        if len(self.classes) == 2:
            self._fit_binary(X, Y)
        else:
            self._fit_multiclass(X, Y)
        return self

    def _fit_binary(self, X, Y):
        n   = len(X)
        rng = LCG(self.seed)
        Yn  = [float(self.label_map[y]) for y in Y]

        p0  = mean(Yn)
        p0  = max(1e-7, min(1 - 1e-7, p0))
        self.F0_binary = log(p0 / (1.0 - p0))
        F   = [self.F0_binary] * n

        self.trees_binary = []

        for m in range(self.n_estimators):
            P = [sigmoid(f) for f in F]

            R = [Yn[i] - P[i] for i in range(n)]

            if self.subsample < 1.0:
                k   = max(1, int(n * self.subsample))
                idx = sample_with_replacement(n, k, rng)
                Xs, Rs = [X[i] for i in idx], [R[i] for i in idx]
                Ps_idx = idx
            else:
                Xs, Rs, Ps_idx = X, R, list(range(n))

            tree = RegressionTree(self.max_depth, self.min_samples)
            tree.fit(Xs, Rs)

            tree = self._newton_leaves_binary(tree, X, R, P)
            self.trees_binary.append(tree)

            update = tree.predict(X)
            F = [F[i] + self.lr * update[i] for i in range(n)]

    def _newton_leaves_binary(self, tree, X, R, P):

        def update_node(node, X, R, P):
            if node[0] == 'leaf':
                return node
            _, f, t, left, right = node
            li = [i for i in range(len(X)) if X[i][f] <= t]
            ri = [i for i in range(len(X)) if X[i][f] >  t]
            Xl, Rl, Pl = [X[i] for i in li], [R[i] for i in li], [P[i] for i in li]
            Xr, Rr, Pr = [X[i] for i in ri], [R[i] for i in ri], [P[i] for i in ri]
            new_left  = self._newton_leaf_value(Rl, Pl) if not Xl else update_node(left,  Xl, Rl, Pl)
            new_right = self._newton_leaf_value(Rr, Pr) if not Xr else update_node(right, Xr, Rr, Pr)
            return ('node', f, t, new_left, new_right)

        def replace_leaves(node, X, R, P):
            if node[0] == 'leaf':
                denom = sum(p*(1-p) for p in P)
                val   = sum(R) / denom if denom > 1e-12 else 0.0
                return ('leaf', val)
            _, f, t, left, right = node
            li = [i for i in range(len(X)) if X[i][f] <= t]
            ri = [i for i in range(len(X)) if X[i][f] >  t]
            Xl = [X[i] for i in li]; Rl = [R[i] for i in li]; Pl = [P[i] for i in li]
            Xr = [X[i] for i in ri]; Rr = [R[i] for i in ri]; Pr = [P[i] for i in ri]
            return ('node', f, t,
                    replace_leaves(left,  Xl, Rl, Pl),
                    replace_leaves(right, Xr, Rr, Pr))

        tree.root = replace_leaves(tree.root, X, R, P)
        return tree

    def _newton_leaf_value(self, R, P):
        denom = sum(p*(1-p) for p in P)
        val   = sum(R) / denom if denom > 1e-12 else 0.0
        return ('leaf', val)

    def _fit_multiclass(self, X, Y):
        n  = len(X)
        nc = len(self.classes)
        rng = LCG(self.seed)

        Yn = [[1.0 if self.label_map[y] == c else 0.0 for c in range(nc)]
              for y in Y]

        counts = count_values(Y)
        self.F0_multi = []
        for c in self.classes:
            p = counts.get(c, 0) / n
            p = max(1e-7, min(1-1e-7, p))
            self.F0_multi.append(log(p))

        F = [[self.F0_multi[c] for c in range(nc)] for _ in range(n)]

        self.trees_multi = [[] for _ in range(nc)]

        for m in range(self.n_estimators):
            P = [softmax(F[i]) for i in range(n)]

            for c in range(nc):
                R = [Yn[i][c] - P[i][c] for i in range(n)]

                if self.subsample < 1.0:
                    k   = max(1, int(n * self.subsample))
                    idx = sample_with_replacement(n, k, rng)
                    Xs, Rs = [X[i] for i in idx], [R[i] for i in idx]
                else:
                    Xs, Rs = X, R

                tree = RegressionTree(self.max_depth, self.min_samples)
                tree.fit(Xs, Rs)
                self.trees_multi[c].append(tree)

                update = tree.predict(X)
                for i in range(n):
                    F[i][c] += self.lr * update[i]

    def decision_function(self, X):
        if len(self.classes) == 2:
            F = [self.F0_binary] * len(X)
            for tree in self.trees_binary:
                upd = tree.predict(X)
                F = [F[i] + self.lr * upd[i] for i in range(len(X))]
            return F
        else:
            nc = len(self.classes)
            F  = [[self.F0_multi[c] for c in range(nc)] for _ in range(len(X))]
            for c in range(nc):
                for tree in self.trees_multi[c]:
                    upd = tree.predict(X)
                    for i in range(len(X)):
                        F[i][c] += self.lr * upd[i]
            return F

    def predict_proba(self, X):
        scores = self.decision_function(X)
        if len(self.classes) == 2:
            return [[1 - sigmoid(s), sigmoid(s)] for s in scores]
        return [softmax(row) for row in scores]

    def predict(self, X):
        proba = self.predict_proba(X)
        if len(self.classes) == 2:
            return [self.classes[1] if p[1] >= 0.5 else self.classes[0]
                    for p in proba]
        return [self.classes[max(range(len(self.classes)),
                key=lambda c: p[c])] for p in proba]

    def accuracy(self, X, Y):
        return accuracy(Y, self.predict(X))

    def score_report(self, X, Y, name=''):
        preds = self.predict(X)
        tag   = f' [{name}]' if name else ''
        print(f'  Accuracy{tag}: {accuracy(Y, preds)*100:.2f}%')
        per_class_accuracy(Y, preds)


def gen_blobs(n=400, n_classes=3, spread=0.7, seed=1):
    g = LCG(seed)
    centers = [[0,2],[-2,-1],[2,-1],[-2,2],[2,2]][:n_classes]
    X, Y = [], []
    for c, ce in enumerate(centers):
        for _ in range(n//n_classes):
            X.append([ce[0]+g.next_gaussian()*spread,
                      ce[1]+g.next_gaussian()*spread])
            Y.append(c)
    return X, Y

def gen_regression(n=300, seed=5):
    g = LCG(seed); X, Y = [], []
    def sin_a(x):
        r,t=x,x
        for k in range(1,15): t*=-x*x/((2*k)*(2*k+1)); r+=t
        return r
    for _ in range(n):
        x1=g.next_float(-3,3); x2=g.next_float(-3,3)
        X.append([x1,x2])
        Y.append(float(sin_a(x1)+x2**2*0.3+g.next_gaussian()*0.3))
    return X, Y

sep = '='*52

print(sep)
print('TEST 1 — GBM Regressor | Sine surface')
print(sep)
X, Y = gen_regression(400)
Xtr,Xte,Ytr,Yte = train_test_split(X, Y)
print()
for n_est in [10, 50, 100, 200]:
    gbr = GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.1,
                                     max_depth=3, seed=42)
    gbr.fit(Xtr, Ytr)
    gbr.score_report(Xte, Yte, name=f'n={n_est}')

print()
print(sep)
print('TEST 2  Learning rate vs n_estimators tradeoff')
print('  Small lr needs more trees. Large lr overfits fast.')
print(sep)
print()
for lr, n_est in [(0.5,20),(0.1,100),(0.05,200),(0.01,500)]:
    gbr = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr,
                                     max_depth=3, seed=42)
    gbr.fit(Xtr, Ytr)
    preds = gbr.predict(Xte)
    mse = mean_squared_error([float(y) for y in Yte], preds)
    print(f'  lr={lr:<5}  n={n_est:<4}  MSE={mse:.4f}')

print()
print(sep)
print('TEST 3 GBM Binary Classifier | 2-class blobs')
print(sep)
X2, Y2 = gen_blobs(400, 2, spread=0.9)
Xtr2,Xte2,Ytr2,Yte2 = train_test_split(X2, Y2)
print()
gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                  max_depth=3, seed=42)
gbc.fit(Xtr2, Ytr2)
gbc.score_report(Xte2, Yte2, name='Binary GBM')

print()
print(sep)
print('TEST 4 GBM Multiclass Classifier | 3-class blobs')
print(sep)
X3, Y3 = gen_blobs(450, 3, spread=0.8)
Xtr3,Xte3,Ytr3,Yte3 = train_test_split(X3, Y3)
print()
gbc3 = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                   max_depth=3, seed=42)
gbc3.fit(Xtr3, Ytr3)
gbc3.score_report(Xte3, Yte3, name='Multiclass GBM')

print()
print(sep)
print('TEST 5 GBM vs Random Forest vs Single DT')
print(sep)
print()
from random_forest import (
    RandomForestClassifier as RFC,
    DecisionTree as DT,
)
for name, model in [
    ('Decision Tree',   DT(max_depth=5)),
    ('Random Forest',   RFC(n_trees=50, max_depth=5, seed=42)),
    ('Gradient Boost',  GradientBoostingClassifier(n_estimators=50,
                            learning_rate=0.1, max_depth=3, seed=42)),
]:
    model.fit(Xtr3, Ytr3)
    acc = accuracy(Yte3, model.predict(Xte3))
    print(f'  {name:<18}: {acc*100:.2f}%')

print()
print(sep)
print('TEST 6 Subsample effect (Stochastic GB)')
print('  subsample < 1.0 adds noise, reduces overfitting')
print(sep)
print()
for sub in [1.0, 0.8, 0.6, 0.5]:
    gbr_s = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, subsample=sub, seed=42)
    gbr_s.fit(Xtr, Ytr)
    preds = gbr_s.predict(Xte)
    mse = mean_squared_error([float(y) for y in Yte], preds)
    print(f'  subsample={sub}  MSE={mse:.4f}')
