from utilities import (
    mean, variance, entropy, gini_impurity, log2,
    LCG, train_test_split, unique, count_values,
    accuracy, per_class_accuracy, confusion_matrix,
    mean_squared_error, mean_absolute_error,
    sample_with_replacement, random_subset, argmax
)

rng = LCG(42)

class Node:
    def __init__(self):
        self.feature    = None
        self.threshold  = None
        self.left       = None
        self.right      = None
        self.leaf_value = None

    def is_leaf(self): return self.leaf_value is not None

def split(X, Y, feature, threshold):
    l_idx = [i for i in range(len(X)) if X[i][feature] <= threshold]
    r_idx = [i for i in range(len(X)) if X[i][feature] >  threshold]
    return l_idx, r_idx

def gini(Y):
    counts = list(count_values(Y).values())
    return gini_impurity(counts)

def mse_impurity(Y):
    if len(Y) == 0: return 0.0
    m = mean(Y)
    return mean([(y - m) ** 2 for y in Y])

def best_split(X, Y, feature_indices, criterion):
    best_gain, best_feat, best_thresh = -1, None, None
    impurity_fn = gini if criterion == 'gini' else mse_impurity
    parent_imp  = impurity_fn(Y)
    n           = len(Y)

    for f in feature_indices:
        thresholds = sorted(set(x[f] for x in X))
        for i in range(len(thresholds) - 1):
            thresh        = (thresholds[i] + thresholds[i+1]) / 2
            l_idx, r_idx  = split(X, Y, f, thresh)
            if len(l_idx) == 0 or len(r_idx) == 0: continue
            Yl = [Y[i] for i in l_idx]
            Yr = [Y[i] for i in r_idx]
            gain = parent_imp - (len(Yl)/n * impurity_fn(Yl)
                               + len(Yr)/n * impurity_fn(Yr))
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, f, thresh

    return best_feat, best_thresh

class DecisionTree:
    def __init__(self, max_depth=None, min_samples=2,
                 criterion='gini', max_features=None, rng=None):
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.criterion   = criterion
        self.max_features = max_features
        self.rng         = rng or LCG(0)
        self.root        = None
        self.task        = None

    def fit(self, X, Y):
        self.task = 'regression' if isinstance(Y[0], float) else 'classification'
        self.n_features = len(X[0])
        self.root = self._build(X, Y, depth=0)
        return self

    def _build(self, X, Y, depth):
        node = Node()

        pure     = len(unique(Y)) == 1
        too_small = len(Y) < self.min_samples
        too_deep  = self.max_depth is not None and depth >= self.max_depth

        if pure or too_small or too_deep:
            node.leaf_value = self._leaf(Y)
            return node

        all_feats = list(range(self.n_features))
        if self.max_features is not None:
            k = min(self.max_features, self.n_features)
            feat_subset = random_subset(all_feats, k, self.rng)
        else:
            feat_subset = all_feats

        f, thresh = best_split(X, Y, feat_subset, self.criterion)

        if f is None:
            node.leaf_value = self._leaf(Y)
            return node

        l_idx, r_idx = split(X, Y, f, thresh)
        node.feature   = f
        node.threshold = thresh
        node.left  = self._build([X[i] for i in l_idx],
                                  [Y[i] for i in l_idx], depth+1)
        node.right = self._build([X[i] for i in r_idx],
                                  [Y[i] for i in r_idx], depth+1)
        return node

    def _leaf(self, Y):
        if self.task == 'regression':
            return mean(Y)
        counts = count_values(Y)
        return max(counts, key=lambda k: counts[k])

    def _predict_one(self, x, node):
        if node.is_leaf(): return node.leaf_value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return [self._predict_one(x, self.root) for x in X]


class RandomForestClassifier:
    def __init__(self, n_trees=100, max_depth=None, min_samples=2,
                 max_features='sqrt', criterion='gini', seed=42):
        self.n_trees     = n_trees
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.criterion   = criterion
        self.seed        = seed
        self.trees       = []
        self.max_features = max_features

    def _resolve_max_features(self, n_features):
        if self.max_features == 'sqrt': return max(1, int(n_features ** 0.5))
        if self.max_features == 'log2': return max(1, int(log2(n_features)))
        if isinstance(self.max_features, int): return self.max_features
        return n_features

    def fit(self, X, Y):
        n            = len(X)
        n_features   = len(X[0])
        max_feat     = self._resolve_max_features(n_features)
        self.classes = sorted(unique(Y))
        self.trees   = []

        for t in range(self.n_trees):
            tree_rng = LCG(self.seed + t)
            idx      = sample_with_replacement(n, n, tree_rng)
            Xb       = [X[i] for i in idx]
            Yb       = [Y[i] for i in idx]
            tree     = DecisionTree(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                criterion=self.criterion,
                max_features=max_feat,
                rng=tree_rng
            )
            tree.fit(Xb, Yb)
            self.trees.append(tree)
        return self

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        final = []
        for i in range(len(X)):
            votes  = [all_preds[t][i] for t in range(self.n_trees)]
            counts = count_values(votes)
            final.append(max(counts, key=lambda k: counts[k]))
        return final

    def predict_proba(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        result = []
        for i in range(len(X)):
            votes  = [all_preds[t][i] for t in range(self.n_trees)]
            counts = count_values(votes)
            total  = sum(counts.values())
            result.append({c: counts.get(c, 0) / total for c in self.classes})
        return result

    def accuracy(self, X, Y):   return accuracy(Y, self.predict(X))

    def score_report(self, X, Y, name=''):
        preds = self.predict(X)
        acc   = accuracy(Y, preds)
        tag   = f' [{name}]' if name else ''
        print(f'  Accuracy{tag}: {acc*100:.2f}%')
        per_class_accuracy(Y, preds)
        cm, classes = confusion_matrix(Y, preds)
        print(f'  Confusion matrix:')
        print('        ' + '  '.join(f'{c:>6}' for c in classes))
        for i, row in enumerate(cm):
            print(f'  {classes[i]:>6}  ' + '  '.join(f'{v:>6}' for v in row))

    def feature_importance(self, n_features, feature_names=None):
        counts = [0] * n_features
        total  = 0
        def walk(node):
            nonlocal total
            if node is None or node.is_leaf(): return
            counts[node.feature] += 1
            total += 1
            walk(node.left)
            walk(node.right)
        for tree in self.trees:
            walk(tree.root)
        if total == 0: return
        print('  Feature importances (split frequency):')
        names = feature_names or [f'f{i}' for i in range(n_features)]
        pairs = sorted(zip(names, counts), key=lambda x: -x[1])
        for name, c in pairs:
            bar = '█' * int(30 * c / total)
            print(f'  {name:>12}  {c/total*100:>5.1f}%  {bar}')


class RandomForestRegressor:
    def __init__(self, n_trees=100, max_depth=None, min_samples=2,
                 max_features='sqrt', seed=42):
        self.n_trees     = n_trees
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.seed        = seed
        self.trees       = []

    def _resolve_max_features(self, n_features):
        if self.max_features == 'sqrt': return max(1, int(n_features ** 0.5))
        if self.max_features == 'log2': return max(1, int(log2(n_features)))
        if isinstance(self.max_features, int): return self.max_features
        return n_features

    def fit(self, X, Y):
        n          = len(X)
        n_features = len(X[0])
        max_feat   = self._resolve_max_features(n_features)
        self.trees = []

        for t in range(self.n_trees):
            tree_rng = LCG(self.seed + t)
            idx      = sample_with_replacement(n, n, tree_rng)
            Xb       = [X[i] for i in idx]
            Yb       = [float(Y[i]) for i in idx]
            tree     = DecisionTree(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                criterion='mse',
                max_features=max_feat,
                rng=tree_rng
            )
            tree.fit(Xb, Yb)
            self.trees.append(tree)
        return self

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        return [mean([all_preds[t][i] for t in range(self.n_trees)])
                for i in range(len(X))]

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
            X.append([ce[0]+g.next_gaussian()*0.6,
                      ce[1]+g.next_gaussian()*0.6])
            Y.append(c)
    return X, Y

def gen_circles(n=400, seed=3):
    from utilities import PI, cos, sin
    g = LCG(seed); X, Y = [], []
    def cos_a(x):
        r,t=1.0,1.0
        for k in range(1,15):
            t*=-x*x/((2*k-1)*(2*k)); r+=t
        return r
    def sin_a(x):
        r,t=x,x
        for k in range(1,15):
            t*=-x*x/((2*k)*(2*k+1)); r+=t
        return r
    for i in range(n):
        a = g.next_float(0, 2*PI)
        r, y = (0.5+g.next_gaussian()*0.07, 0) if i%2==0 \
               else (1.5+g.next_gaussian()*0.07, 1)
        X.append([r*cos_a(a), r*sin_a(a)]); Y.append(y)
    return X, Y

def gen_regression(n=300, seed=5):
    from utilities import PI
    g = LCG(seed); X, Y = [], []
    def sin_a(x):
        r,t=x,x
        for k in range(1,15):
            t*=-x*x/((2*k)*(2*k+1)); r+=t
        return r
    for _ in range(n):
        x1 = g.next_float(-3, 3)
        x2 = g.next_float(-3, 3)
        y  = sin_a(x1) + x2**2 * 0.3 + g.next_gaussian()*0.2
        X.append([x1, x2]); Y.append(float(y))
    return X, Y


sep = '='*52

print(sep)
print('TEST 1 — RF Classifier | Blobs (3 class)')
print(sep)
X, Y = gen_blobs(450, 3)
Xtr,Xte,Ytr,Yte = train_test_split(X, Y)
rf = RandomForestClassifier(n_trees=50, max_depth=6, seed=42)
rf.fit(Xtr, Ytr)
rf.score_report(Xte, Yte, name='50 trees')
print()
rf.feature_importance(2, ['x1','x2'])

print()
print(sep)
print('TEST 2 — RF Classifier | Circles (non-linear)')
print(sep)
X, Y = gen_circles(400)
Xtr,Xte,Ytr,Yte = train_test_split(X, Y)
rf2 = RandomForestClassifier(n_trees=50, max_depth=8, seed=42)
rf2.fit(Xtr, Ytr)
rf2.score_report(Xte, Yte, name='circles')

print()
print(sep)
print('TEST 3 — n_trees effect')
print(sep)
X, Y = gen_blobs(450, 3)
Xtr,Xte,Ytr,Yte = train_test_split(X, Y)
for n in [1, 5, 10, 25, 50, 100]:
    rf_n = RandomForestClassifier(n_trees=n, max_depth=6, seed=42)
    rf_n.fit(Xtr, Ytr)
    acc = rf_n.accuracy(Xte, Yte)
    print(f'  n_trees={n:>4}  acc={acc*100:.2f}%')

print()
print(sep)
print('TEST 4 — Single DT vs Random Forest')
print(sep)
X, Y = gen_blobs(450, 3)
Xtr,Xte,Ytr,Yte = train_test_split(X, Y)
dt = DecisionTree(max_depth=6)
dt.fit(Xtr, Ytr)
dt_acc = accuracy(Yte, dt.predict(Xte))
rf_s = RandomForestClassifier(n_trees=50, max_depth=6, seed=42)
rf_s.fit(Xtr, Ytr)
rf_acc = rf_s.accuracy(Xte, Yte)
print(f'  Single Decision Tree : {dt_acc*100:.2f}%')
print(f'  Random Forest 50     : {rf_acc*100:.2f}%')

print()
print(sep)
print('TEST 5 — RF Regressor')
print(sep)
X, Y = gen_regression(300)
Xtr,Xte,Ytr,Yte = train_test_split(X, Y)
rfr = RandomForestRegressor(n_trees=50, max_depth=6, seed=42)
rfr.fit(Xtr, Ytr)
rfr.score_report(Xte, Yte, name='RF Regressor')
