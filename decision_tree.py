from utilities import (
    entropy, gini_impurity, mean, count_values, unique,
    train_test_split, accuracy, per_class_accuracy, confusion_matrix,
    variance, LCG
)



class Node:
    def __init__(self):
        # split info
        self.feature   = None
        self.threshold = None
        self.left      = None
        self.right     = None
        # leaf info
        self.is_leaf   = False
        self.prediction = None

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(pred={self.prediction})"
        return f"Split(feat={self.feature}, thresh={self.threshold:.4f})"



def _class_counts(Y):
    return list(count_values(Y).values())

def _impurity(Y, criterion):
    counts = _class_counts(Y)
    if criterion == "entropy":
        return entropy(counts)
    elif criterion == "gini":
        return gini_impurity(counts)
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use 'entropy' or 'gini'.")

def _variance_reduction(Y):
    return variance(Y)

def _information_gain(Y, Y_left, Y_right, criterion):
    n  = len(Y)
    nl = len(Y_left)
    nr = len(Y_right)
    if nl == 0 or nr == 0:
        return 0.0
    parent_imp = _impurity(Y, criterion)
    child_imp  = (nl / n) * _impurity(Y_left,  criterion) \
               + (nr / n) * _impurity(Y_right, criterion)
    return parent_imp - child_imp

def _variance_gain(Y, Y_left, Y_right):
    n  = len(Y)
    nl = len(Y_left)
    nr = len(Y_right)
    if nl == 0 or nr == 0:
        return 0.0
    return _variance_reduction(Y) \
         - (nl / n) * _variance_reduction(Y_left) \
         - (nr / n) * _variance_reduction(Y_right)




def _split(X, Y, feature, threshold):
    left_X,  left_Y  = [], []
    right_X, right_Y = [], []
    for row, label in zip(X, Y):
        if row[feature] <= threshold:
            left_X.append(row);  left_Y.append(label)
        else:
            right_X.append(row); right_Y.append(label)
    return left_X, left_Y, right_X, right_Y

def _best_split(X, Y, criterion, task, max_features=None):
    n_features  = len(X[0])
    best_gain   = -1.0
    best_feat   = None
    best_thresh = None

    feature_indices = list(range(n_features))

    if max_features is not None and max_features < n_features:
        step = n_features // max_features
        feature_indices = [i * step % n_features for i in range(max_features)]
        feature_indices = list(set(feature_indices))  # deduplicate

    for f in feature_indices:
        col_vals = sorted(set(row[f] for row in X))
        thresholds = [(col_vals[i] + col_vals[i+1]) / 2
                      for i in range(len(col_vals) - 1)]

        for thresh in thresholds:
            _, ly, _, ry = _split(X, Y, f, thresh)
            if len(ly) == 0 or len(ry) == 0:
                continue

            if task == "classification":
                gain = _information_gain(Y, ly, ry, criterion)
            else:
                gain = _variance_gain(Y, ly, ry)

            if gain > best_gain:
                best_gain   = gain
                best_feat   = f
                best_thresh = thresh

    return best_feat, best_thresh, best_gain



class DecisionTree:

    def __init__(self, criterion="gini", task="classification",
                 max_depth=None, min_samples=2, max_features=None):
        self.criterion    = criterion
        self.task         = task
        self.max_depth    = max_depth
        self.min_samples  = min_samples
        self.max_features = max_features
        self.root         = None


    def fit(self, X, Y):
        self.root = self._build(X, Y, depth=0)

    def _build(self, X, Y, depth):
        node = Node()

        pure        = len(set(Y)) == 1
        too_small   = len(Y) < self.min_samples
        max_reached = (self.max_depth is not None) and (depth >= self.max_depth)

        if pure or too_small or max_reached:
            node.is_leaf    = True
            node.prediction = self._leaf_value(Y)
            return node

        feat, thresh, gain = _best_split(
            X, Y, self.criterion, self.task, self.max_features
        )

        if feat is None or gain <= 0:
            node.is_leaf    = True
            node.prediction = self._leaf_value(Y)
            return node

        lX, lY, rX, rY = _split(X, Y, feat, thresh)

        node.feature   = feat
        node.threshold = thresh
        node.left      = self._build(lX, lY, depth + 1)
        node.right     = self._build(rX, rY, depth + 1)
        return node

    def _leaf_value(self, Y):
        if self.task == "classification":
            return max(count_values(Y), key=lambda k: count_values(Y)[k])
        else:
            return mean(Y)


    def predict_one(self, row):
        node = self.root
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X):
        return [self.predict_one(row) for row in X]

    def accuracy(self, X, Y):
        return accuracy(Y, self.predict(X))


    def depth(self):
        return self._depth(self.root)

    def _depth(self, node):
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def n_leaves(self):
        return self._count_leaves(self.root)

    def _count_leaves(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def print_tree(self, node=None, indent="", last=True):
        if node is None:
            node = self.root
        branch = "└── " if last else "├── "
        if node.is_leaf:
            print(indent + branch + f"LEAF → {node.prediction}")
        else:
            print(indent + branch +
                  f"feat[{node.feature}] <= {node.threshold:.4f} ?")
            child_indent = indent + ("    " if last else "│   ")
            self.print_tree(node.left,  child_indent, last=False)
            self.print_tree(node.right, child_indent, last=True)

    def feature_importances(self, n_features):
        importances = [0.0] * n_features
        self._collect_importances(self.root, importances)
        total = sum(importances)
        if total == 0:
            return importances
        return [v / total for v in importances]

    def _collect_importances(self, node, importances):
        if node is None or node.is_leaf:
            return
        importances[node.feature] += 1          # simple count; extend as needed
        self._collect_importances(node.left,  importances)
        self._collect_importances(node.right, importances)



def generate_blobs(n_per_class=100, seed=0):
    rng = LCG(seed)
    X, Y = [], []
    centers = [(1.0, 1.0, 0), (5.0, 1.0, 1), (3.0, 5.0, 2)]
    for cx, cy, label in centers:
        for _ in range(n_per_class):
            X.append([cx + rng.next_gaussian() * 0.7,
                      cy + rng.next_gaussian() * 0.7])
            Y.append(label)
    return X, Y

def generate_xor(n=200, seed=1):
    rng = LCG(seed)
    X, Y = [], []
    for _ in range(n):
        x1 = rng.next_float(-2, 2)
        x2 = rng.next_float(-2, 2)
        label = 1 if (x1 * x2 > 0) else 0
        X.append([x1, x2])
        Y.append(label)
    return X, Y

def generate_regression(n=200, seed=2):
    rng = LCG(seed)
    X, Y = [], []
    for _ in range(n):
        x = rng.next_float(-5, 5)
        y = 3 * x + rng.next_gaussian() * 2.0
        X.append([x])
        Y.append(y)
    return X, Y



def print_cm(cm, classes):
    header = "       " + "  ".join(f"{c:>4}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"  act {classes[i]}  " + "  ".join(f"{v:>4}" for v in row))


print("=" * 50)
print("TEST 1 — Classification (Gini, blobs)")
print("=" * 50)
X, Y = generate_blobs(n_per_class=120)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_ratio=0.2)
tree = DecisionTree(criterion="gini", max_depth=6)
tree.fit(Xtr, Ytr)
preds = tree.predict(Xte)
print(f"  Accuracy  : {accuracy(Yte, preds):.2%}")
print(f"  Depth     : {tree.depth()}")
print(f"  Leaves    : {tree.n_leaves()}")
print("  Per-class :")
per_class_accuracy(Yte, preds)
print("  Confusion matrix (pred →):")
cm, cls = confusion_matrix(Yte, preds)
print_cm(cm, cls)
print()
print("  Tree structure (first 3 levels):")
tree2 = DecisionTree(criterion="gini", max_depth=3)
tree2.fit(Xtr, Ytr)
tree2.print_tree()

print()
print("=" * 50)
print("TEST 2 — Classification (Entropy, XOR)")
print("=" * 50)
X, Y = generate_xor(n=400)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_ratio=0.2)
tree = DecisionTree(criterion="entropy", max_depth=8)
tree.fit(Xtr, Ytr)
preds = tree.predict(Xte)
print(f"  Accuracy : {accuracy(Yte, preds):.2%}")
print(f"  Depth    : {tree.depth()}")
print(f"  Leaves   : {tree.n_leaves()}")
print("  Per-class:")
per_class_accuracy(Yte, preds)

print()
print("=" * 50)
print("TEST 3 — Regression (variance reduction)")
print("=" * 50)
from utilities import mean_squared_error, mean_absolute_error, sqrt
X, Y = generate_regression(n=300)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_ratio=0.2)
tree = DecisionTree(task="regression", max_depth=6)
tree.fit(Xtr, Ytr)
preds = tree.predict(Xte)
mse = mean_squared_error(Yte, preds)
mae = mean_absolute_error(Yte, preds)
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {sqrt(mse):.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  Depth: {tree.depth()}")

print()
print("=" * 50)
print("TEST 4 — Overfitting: full depth vs pruned")
print("=" * 50)
X, Y = generate_blobs(n_per_class=80)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_ratio=0.25)
for depth in [1, 2, 3, 5, None]:
    t = DecisionTree(criterion="gini", max_depth=depth)
    t.fit(Xtr, Ytr)
    tr_acc = accuracy(Ytr, t.predict(Xtr))
    te_acc = accuracy(Yte, t.predict(Xte))
    label  = str(depth) if depth else "∞"
    print(f"  max_depth={label:>3}  train={tr_acc:.2%}  test={te_acc:.2%}"
          f"  leaves={t.n_leaves()}")

print()
print("done.")