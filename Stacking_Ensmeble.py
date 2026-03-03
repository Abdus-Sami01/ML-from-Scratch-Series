from utilities import (
    mean, variance, std, log2, sqrt, absolute,
    LCG, train_test_split, unique, count_values,
    accuracy, per_class_accuracy, confusion_matrix,
    mean_squared_error, mean_absolute_error,
    sample_with_replacement, random_subset,
    cross_val_split, argmax
)


def clone(model):
    import inspect
    try:
        sig = inspect.signature(model.__class__.__init__)
        keys = set(list(sig.parameters.keys())[1:])
    except Exception:
        keys = set(model.__dict__.keys())
    params = {k: v for k, v in model.__dict__.items() if k in keys}
    return model.__class__(**params)

rng = LCG(42)


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, Y):
        self.X, self.Y = X, Y
        self.classes = sorted(unique(Y))
        return self

    def _pred_one(self, x):
        dists = sorted([(sum((a-b)**2 for a,b in zip(x,xt))**0.5, i)
                        for i, xt in enumerate(self.X)])[:self.k]
        counts = {}
        for _, i in dists:
            counts[self.Y[i]] = counts.get(self.Y[i], 0) + 1
        return max(counts, key=lambda c: counts[c])

    def predict(self, X): return [self._pred_one(x) for x in X]


class NaiveBayes:
    def fit(self, X, Y):
        self.classes = sorted(unique(Y))
        n = len(Y)
        self.priors, self.means, self.vars = {}, {}, {}
        for c in self.classes:
            idx = [i for i,y in enumerate(Y) if y == c]
            self.priors[c] = len(idx) / n
            self.means[c]  = [mean([X[i][j] for i in idx])
                               for j in range(len(X[0]))]
            self.vars[c]   = [variance([X[i][j] for i in idx]) + 1e-9
                               for j in range(len(X[0]))]
        return self

    def _score(self, x, c):
        import math
        s = math.log(self.priors[c])
        for j in range(len(x)):
            mu, var = self.means[c][j], self.vars[c][j]
            s += -0.5 * math.log(2 * 3.141592653589793 * var)
            s += -((x[j] - mu) ** 2) / (2 * var)
        return s

    def predict(self, X):
        return [max(self.classes, key=lambda c: self._score(x, c))
                for x in X]


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=200):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, Y):
        self.classes = sorted(unique(Y))
        self.label_map = {c: i for i, c in enumerate(self.classes)}
        n, nf = len(X), len(X[0])
        self.W = [[0.0]*nf for _ in range(len(self.classes))]
        self.b = [0.0] * len(self.classes)
        Yn = [self.label_map[y] for y in Y]
        for _ in range(self.epochs):
            for xi, yi in zip(X, Yn):
                scores = [sum(self.W[c][j]*xi[j]
                          for j in range(nf)) + self.b[c]
                          for c in range(len(self.classes))]
                probs  = self._softmax(scores)
                for c in range(len(self.classes)):
                    err = probs[c] - (1.0 if c == yi else 0.0)
                    for j in range(nf):
                        self.W[c][j] -= self.lr * err * xi[j]
                    self.b[c] -= self.lr * err
        return self

    def _softmax(self, scores):
        m = max(scores)
        e = [2.718281828459045 ** (s - m) for s in scores]
        t = sum(e)
        return [v/t for v in e]

    def predict(self, X):
        results = []
        nf = len(X[0])
        for xi in X:
            scores = [sum(self.W[c][j]*xi[j]
                      for j in range(nf)) + self.b[c]
                      for c in range(len(self.classes))]
            results.append(self.classes[scores.index(max(scores))])
        return results


class DecisionTree:
    def __init__(self, max_depth=4, min_samples=2):
        self.max_depth, self.min_samples = max_depth, min_samples

    def fit(self, X, Y):
        self.task = 'regression' if isinstance(Y[0], float) else 'classification'
        self.nf   = len(X[0])
        self.root = self._build(X, Y, 0)
        return self

    def _gini(self, Y):
        counts = list(count_values(Y).values())
        n = sum(counts)
        return 1 - sum((c/n)**2 for c in counts) if n else 0.0

    def _mse(self, Y):
        if not Y: return 0.0
        m = mean(Y); return mean([(y-m)**2 for y in Y])

    def _best_split(self, X, Y):
        imp_fn = self._gini if self.task == 'classification' else self._mse
        best   = (-1, None, None)
        parent = imp_fn(Y)
        n      = len(Y)
        for f in range(self.nf):
            thresholds = sorted(set(x[f] for x in X))
            for i in range(len(thresholds)-1):
                t   = (thresholds[i] + thresholds[i+1]) / 2
                li  = [k for k in range(n) if X[k][f] <= t]
                ri  = [k for k in range(n) if X[k][f] >  t]
                if not li or not ri: continue
                Yl, Yr = [Y[k] for k in li], [Y[k] for k in ri]
                gain = parent - (len(Yl)/n*imp_fn(Yl) + len(Yr)/n*imp_fn(Yr))
                if gain > best[0]: best = (gain, f, t)
        return best[1], best[2]

    def _leaf(self, Y):
        if self.task == 'regression': return mean(Y)
        c = count_values(Y); return max(c, key=lambda k: c[k])

    def _build(self, X, Y, d):
        if len(unique(Y))==1 or len(Y)<self.min_samples or d>=self.max_depth:
            return ('leaf', self._leaf(Y))
        f, t = self._best_split(X, Y)
        if f is None: return ('leaf', self._leaf(Y))
        li = [i for i in range(len(X)) if X[i][f] <= t]
        ri = [i for i in range(len(X)) if X[i][f] >  t]
        return ('node', f, t,
                self._build([X[i] for i in li],[Y[i] for i in li], d+1),
                self._build([X[i] for i in ri],[Y[i] for i in ri], d+1))

    def _pred_one(self, x, node):
        if node[0] == 'leaf': return node[1]
        _, f, t, left, right = node
        return self._pred_one(x, left if x[f] <= t else right)

    def predict(self, X): return [self._pred_one(x, self.root) for x in X]


class RandomForest:
    def __init__(self, n_trees=30, max_depth=5, seed=42):
        self.n_trees, self.max_depth, self.seed = n_trees, max_depth, seed
        self.trees = []

    def fit(self, X, Y):
        self.task = 'regression' if isinstance(Y[0], float) else 'classification'
        n = len(X)
        for t in range(self.n_trees):
            tr = LCG(self.seed + t)
            idx = sample_with_replacement(n, n, tr)
            Xb, Yb = [X[i] for i in idx], [Y[i] for i in idx]
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(Xb, Yb)
            self.trees.append(tree)
        return self

    def predict(self, X):
        all_p = [t.predict(X) for t in self.trees]
        result = []
        for i in range(len(X)):
            votes = [all_p[t][i] for t in range(self.n_trees)]
            if self.task == 'regression':
                result.append(mean(votes))
            else:
                c = count_values(votes)
                result.append(max(c, key=lambda k: c[k]))
        return result


class LinearMetaLearner:
    def __init__(self, lr=0.05, epochs=500):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, Y):
        nf      = len(X[0])
        self.w  = [0.0] * nf
        self.b  = 0.0
        Yf      = [float(y) for y in Y]
        for _ in range(self.epochs):
            for xi, yi in zip(X, Yf):
                pred = sum(self.w[j]*xi[j] for j in range(nf)) + self.b
                err  = pred - yi
                for j in range(nf):
                    self.w[j] -= self.lr * err * xi[j]
                self.b -= self.lr * err
        return self

    def predict(self, X):
        return [round(sum(self.w[j]*x[j] for j in range(len(x))) + self.b)
                for x in X]


class LogisticMetaLearner:
    def __init__(self, lr=0.1, epochs=300):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, Y):
        self.classes  = sorted(unique(Y))
        self.label_map = {c: i for i, c in enumerate(self.classes)}
        nf, nc = len(X[0]), len(self.classes)
        self.W = [[0.0]*nf for _ in range(nc)]
        self.b = [0.0]*nc
        Yn = [self.label_map[y] for y in Y]
        E  = 2.718281828459045
        for _ in range(self.epochs):
            for xi, yi in zip(X, Yn):
                scores = [sum(self.W[c][j]*xi[j] for j in range(nf))+self.b[c]
                          for c in range(nc)]
                m = max(scores)
                e = [E**(s-m) for s in scores]; t = sum(e)
                probs = [v/t for v in e]
                for c in range(nc):
                    err = probs[c] - (1.0 if c==yi else 0.0)
                    for j in range(nf):
                        self.W[c][j] -= self.lr * err * xi[j]
                    self.b[c] -= self.lr * err
        return self

    def predict(self, X):
        nf = len(X[0])
        E  = 2.718281828459045
        out = []
        for xi in X:
            scores = [sum(self.W[c][j]*xi[j] for j in range(nf))+self.b[c]
                      for c in range(len(self.classes))]
            out.append(self.classes[scores.index(max(scores))])
        return out

class StackingClassifier:

    def __init__(self, base_models, meta_model, n_folds=5, seed=42):
        self.base_models = base_models
        self.meta_model  = meta_model
        self.n_folds     = n_folds
        self.seed        = seed
        self.fitted_bases = []

    def fit(self, X, Y):
        n          = len(X)
        n_base     = len(self.base_models)
        self.classes = sorted(unique(Y))

        self.label_map = {c: i for i, c in enumerate(self.classes)}
        self.inv_map   = {i: c for c, i in self.label_map.items()}

        oof_preds = [[0] * n_base for _ in range(n)]

        folds = cross_val_split(X, Y, self.n_folds, self.seed)

        for fold_idx, (Xtr, Ytr, Xval, Yval) in enumerate(folds):
            val_indices = self._val_indices(n, fold_idx)
            for m_idx, model_template in enumerate(self.base_models):
                model = clone(model_template)
                model.fit(Xtr, Ytr)
                preds = model.predict(Xval)
                for i, pred in zip(val_indices, preds):
                    oof_preds[i][m_idx] = self.label_map.get(pred, pred)

        self.meta_model.fit(oof_preds, Y)

        self.fitted_bases = []
        for model_template in self.base_models:
            model = clone(model_template)
            model.fit(X, Y)
            self.fitted_bases.append(model)

        return self

    def _val_indices(self, n, fold_idx):
        rng       = LCG(self.seed)
        indices   = rng.shuffle_indices(n)
        fold_size = n // self.n_folds
        return indices[fold_idx*fold_size : (fold_idx+1)*fold_size]




    def predict(self, X):
        meta_X = []
        for i in range(len(X)):
            row = []
            for model in self.fitted_bases:
                pred = model.predict([X[i]])[0]
                row.append(self.label_map.get(pred, pred))
            meta_X.append(row)
        return self.meta_model.predict(meta_X)

    def accuracy(self, X, Y): return accuracy(Y, self.predict(X))

    def score_report(self, X, Y, name=''):
        preds = self.predict(X)
        acc   = accuracy(Y, preds)
        tag   = f' [{name}]' if name else ''
        print(f'  Accuracy{tag}: {acc*100:.2f}%')
        per_class_accuracy(Y, preds)


class StackingRegressor:
    def __init__(self, base_models, meta_model, n_folds=5, seed=42):
        self.base_models  = base_models
        self.meta_model   = meta_model
        self.n_folds      = n_folds
        self.seed         = seed
        self.fitted_bases = []

    def fit(self, X, Y):
        n      = len(X)
        n_base = len(self.base_models)
        oof    = [[0.0]*n_base for _ in range(n)]
        folds  = cross_val_split(X, Y, self.n_folds, self.seed)

        for fold_idx, (Xtr, Ytr, Xval, Yval) in enumerate(folds):
            val_idx = self._val_indices(n, fold_idx)
            for m_idx, template in enumerate(self.base_models):
                model = clone(template)
                model.fit(Xtr, Ytr)
                preds = model.predict(Xval)
                for i, p in zip(val_idx, preds):
                    oof[i][m_idx] = float(p)

        self.meta_model.fit(oof, Y)

        self.fitted_bases = []
        for template in self.base_models:
            model = clone(template)
            model.fit(X, Y)
            self.fitted_bases.append(model)
        return self

    def _val_indices(self, n, fold_idx):
        rng       = LCG(self.seed)
        indices   = rng.shuffle_indices(n)
        fold_size = n // self.n_folds
        return indices[fold_idx*fold_size : (fold_idx+1)*fold_size]

    def _clone(self, model):
        init_keys = set(model.__class__.__init__.__code__.co_varnames[1:])
        params = {k: v for k, v in model.__dict__.items() if k in init_keys}
        return model.__class__(**params)

    def predict(self, X):
        meta_X = [[float(m.predict([x])[0]) for m in self.fitted_bases]
                  for x in X]
        return self.meta_model.predict(meta_X)

    def score_report(self, X, Y, name=''):
        preds = self.predict(X)
        mse   = mean_squared_error(Y, preds)
        mae   = mean_absolute_error(Y, preds)
        ym    = mean(Y)
        ss_res = sum((a-b)**2 for a,b in zip(Y,preds))
        ss_tot = sum((y-ym)**2 for y in Y)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        tag = f' [{name}]' if name else ''
        print(f'  MSE{tag}: {mse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}')

def gen_blobs(n=400, n_classes=3, seed=1):
    g = LCG(seed)
    centers = [[0,2],[-2,-1],[2,-1]][:n_classes]
    X, Y = [], []
    for c, ce in enumerate(centers):
        for _ in range(n//n_classes):
            X.append([ce[0]+g.next_gaussian()*0.7,
                      ce[1]+g.next_gaussian()*0.7])
            Y.append(c)
    return X, Y

def gen_regression(n=300, seed=5):
    g = LCG(seed); X, Y = [], []
    def sin_a(x):
        r,t=x,x
        for k in range(1,15):
            t*=-x*x/((2*k)*(2*k+1)); r+=t
        return r
    for _ in range(n):
        x1 = g.next_float(-3,3)
        x2 = g.next_float(-3,3)
        Y.append(float(sin_a(x1) + x2**2*0.3 + g.next_gaussian()*0.3))
        X.append([x1,x2])
    return X, Y


sep = '='*52

print(sep)
print('TEST 1 — Stacking Classifier | Blobs (3 class)')
print(sep)

X, Y = gen_blobs(450, 3)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, seed=42)

base_models = [
    KNNClassifier(k=3),
    KNNClassifier(k=7),
    NaiveBayes(),
    DecisionTree(max_depth=3),
]
meta = LogisticMetaLearner(lr=0.1, epochs=300)

stack_clf = StackingClassifier(base_models, meta, n_folds=5)
stack_clf.fit(Xtr, Ytr)
print()
stack_clf.score_report(Xte, Yte, name='Stacking')

print()
print(sep)
print('TEST 2 — Individual Base Models vs Stacking')
print(sep)
print()

for name, model in [
    ('KNN k=3',        KNNClassifier(k=3)),
    ('KNN k=7',        KNNClassifier(k=7)),
    ('Naive Bayes',    NaiveBayes()),
    ('Decision Tree',  DecisionTree(max_depth=3)),
    ('Random Forest',  RandomForest(n_trees=20, max_depth=4)),
]:
    model.fit(Xtr, Ytr)
    acc = accuracy(Yte, model.predict(Xte))
    print(f'  {name:<18}: {acc*100:.2f}%')

print()
acc_s = stack_clf.accuracy(Xte, Yte)
print(f'  {"Stacking":<18}: {acc_s*100:.2f}%  ← meta learner')

print()
print(sep)
print('TEST 3 — Stacking with 5 base models | Noisy Blobs')
print(sep)

X2, Y2 = gen_blobs(600, 3, seed=99)
Xtr2, Xte2, Ytr2, Yte2 = train_test_split(X2, Y2, seed=7)

base5 = [
    KNNClassifier(k=3),
    KNNClassifier(k=11),
    NaiveBayes(),
    DecisionTree(max_depth=2),
    RandomForest(n_trees=15, max_depth=3),
]
meta2 = LogisticMetaLearner(lr=0.05, epochs=400)
stack2 = StackingClassifier(base5, meta2, n_folds=5)
stack2.fit(Xtr2, Ytr2)
print()
stack2.score_report(Xte2, Yte2, name='5-model stack')

print()
print(sep)
print('TEST 4 — Stacking Regressor')
print(sep)

Xr, Yr = gen_regression(300)
Xrtr, Xrte, Yrtr, Yrte = train_test_split(Xr, Yr, seed=42)

class RidgeMetaLearner:
    def __init__(self, lr=0.01, epochs=500, lam=0.1):
        self.lr, self.epochs, self.lam = lr, epochs, lam
    def fit(self, X, Y):
        nf     = len(X[0])
        self.w = [0.0]*nf
        self.b = 0.0
        Yf     = [float(y) for y in Y]
        for _ in range(self.epochs):
            for xi, yi in zip(X, Yf):
                pred = sum(self.w[j]*xi[j] for j in range(nf)) + self.b
                err  = pred - yi
                for j in range(nf):
                    self.w[j] -= self.lr*(err*xi[j] + self.lam*self.w[j])
                self.b -= self.lr * err
        return self
    def predict(self, X):
        return [sum(self.w[j]*x[j] for j in range(len(x)))+self.b for x in X]

base_reg = [
    DecisionTree(max_depth=3),
    DecisionTree(max_depth=5),
    RandomForest(n_trees=20, max_depth=4),
]

Ytr_f = [float(y) for y in Yrtr]
Yte_f = [float(y) for y in Yrte]

stack_reg = StackingRegressor(base_reg, RidgeMetaLearner(), n_folds=5)
stack_reg.fit(Xrtr, Ytr_f)
print()
stack_reg.score_report(Xrte, Yte_f, name='Stacking Regressor')
