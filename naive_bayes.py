'''
P(class∣features)=P(features)P(features∣class)⋅P(class)​
y_hat​=cargmax​ P(class=c)⋅P(features∣class=c)
'''

from math import log

e = 2.718281828459045

def exp(x):
    res = e ** (x)
    return res

def clip(x , low , high):
    clipped = min(high , max(x, low))
    return clipped

def sign(x):
    sign = 1 if x>= 0 else -1
    return sign

def row_mean(row):
    total = 0
    for v in row:
        total += v
    row_mean = total / len(row)
    return row_mean

def matrix_mean(matrix):
    flatten = []
    for i in matrix:
        for j in i:
            flatten.append(j)
    total = 0
    for k in flatten:
        total += k
    matrix_mean = total/ len(flatten)
    return matrix_mean
     

def variance(x , mean):
    diff = []
    total = 0
    for i in x:
        diff = i - mean
        total += diff ** 2
    if len(x)==0:
     return 1e-9
    var = total/len(x)
    if var == 0:
        return 1e-9
    else:
        return var
    

def sqaure_root(x):
    sqrt= x**0.5
    return sqrt

pi = 3.145

def gaussian_pdf(x , mean , var):
    coeff = 1/sqaure_root(2*pi*var)
    exponent = -((x - mean) ** 2) / (2 * var)
    return coeff * exp(exponent)


class Simple_LCG:
    def __init__(self,seed):
        self.seed = seed
        self.a = 1103515245
        self.c = 12345
        self.m = 2**31

    def next_int(self, low=0, high=100):
        self.seed = (self.a * self.seed + self.c) % self.m
        return low + (self.seed % (high - low + 1))

    def next_float(self, low=0, high=1):
        self.seed = (self.a * self.seed + self.c) % self.m
        return low + (self.seed / self.m) * (high - low)
    
    def next_range(self, lo, hi):
        return lo + self.next_float(0,100) * (hi - lo)

    global cos

    def cos(x, terms=10):
        result = 0
        for n in range(terms):
            sign = (-1) ** n
            numerator = x ** (2 * n)
            denominator = factorial(2 * n)
            result += sign * (numerator / denominator)
        return result


    def next_gaussian(self):
        u1 = self.next_float(0, 1)  # must be (0, 1)
        u2 = self.next_float(0, 1)  # must be (0, 1)
        if u1 == 0:
            u1 = 1e-9
        z = sqaure_root(-2.0 * log(u1)) * cos(2 * pi * u2)
        return z

    global factorial

    def factorial(n):
        res = 1
        for i in range(2, n + 1):
            res *= i
        return res
    


    def shuffle_indices(self, n):
        indices = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.next_int( 0 , i) % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        return indices



def train_test_split(X, Y, test_ratio=0.2, seed=42):
    rng = Simple_LCG(seed)
    n = len(X)
    indices = rng.shuffle_indices(n)

    test_size = int(n * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]
    X_test  = [X[i] for i in test_indices]
    Y_test  = [Y[i] for i in test_indices]

    return X_train, X_test, Y_train, Y_test


class GaussianNB:

    def __init__(self):
        self.classes      = []
        self.class_priors = {}
        self.means        = {}
        self.variances    = {}

    def fit(self, X, Y):
        self.classes = list(set(Y))
        self.classes.sort()

        n_total = len(Y)
        n_features = len(X[0])

        for c in self.classes:
            # grab all rows for this class
            rows = []
            for i in range(len(Y)):
                if Y[i] == c:
                    rows.append(X[i])

            self.class_priors[c] = len(rows) / n_total

            self.means[c]     = []
            self.variances[c] = []

            for f in range(n_features):
                col = [rows[r][f] for r in range(len(rows))]
                m   = row_mean(col)
                v   = variance(col, m)
                self.means[c].append(m)
                self.variances[c].append(v)

    def _log_likelihood(self, x, mean_val, var):
        pdf = gaussian_pdf(x, mean_val, var)
        pdf = clip(pdf, 1e-300, 1e300)
        return log(pdf)

    def predict_scores(self, X):
        all_scores = []
        for row in X:
            scores = {}
            for c in self.classes:
                score = log(self.class_priors[c])
                for f in range(len(row)):
                    score += self._log_likelihood(row[f], self.means[c][f], self.variances[c][f])
                scores[c] = score
            all_scores.append(scores)
        return all_scores

    def predict(self, X):
        scores = self.predict_scores(X)
        predictions = []
        for s in scores:
            best_class = None
            best_score = None
            for c in s:
                if best_score is None or s[c] > best_score:
                    best_score = s[c]
                    best_class = c
            predictions.append(best_class)
        return predictions

    def accuracy(self, X, Y):
        preds = self.predict(X)
        correct = 0
        for i in range(len(Y)):
            if preds[i] == Y[i]:
                correct += 1
        return correct / len(Y)
    
class MultinomialNB:

    def __init__(self):
        self.classes       = []
        self.class_priors  = {}
        self.feature_probs = {}  
    def fit(self, X, Y, alpha=1.0):
        self.classes = list(set(Y))
        self.classes.sort()

        n_total   = len(Y)
        n_features = len(X[0])

        for c in self.classes:
            rows = [X[i] for i in range(len(Y)) if Y[i] == c]

            self.class_priors[c] = len(rows) / n_total

            feature_counts = []
            for f in range(n_features):
                total = 0
                for row in rows:
                    total += row[f]
                feature_counts.append(total)

            total_count = sum(feature_counts) + alpha * n_features

            log_probs = []
            for f in range(n_features):
                p = (feature_counts[f] + alpha) / total_count
                log_probs.append(log(p))

            self.feature_probs[c] = log_probs

    def predict_scores(self, X):
        all_scores = []
        for row in X:
            scores = {}
            for c in self.classes:
                score = log(self.class_priors[c])
                for f in range(len(row)):
                    score += row[f] * self.feature_probs[c][f]
                scores[c] = score
            all_scores.append(scores)
        return all_scores

    def predict(self, X):
        scores = self.predict_scores(X)
        predictions = []
        for s in scores:
            best_class = None
            best_score = None
            for c in s:
                if best_score is None or s[c] > best_score:
                    best_score = s[c]
                    best_class = c
            predictions.append(best_class)
        return predictions

    def accuracy(self, X, Y):
        preds = self.predict(X)
        correct = 0
        for i in range(len(Y)):
            if preds[i] == Y[i]:
                correct += 1
        return correct / len(Y)
    

class BernoulliNB:

    def __init__(self):
        self.classes       = []
        self.class_priors  = {}
        self.feature_probs = {}   # P(feature=1 | class)

    def fit(self, X, Y, alpha=1.0):
        self.classes = list(set(Y))
        self.classes.sort()

        n_total    = len(Y)
        n_features = len(X[0])

        for c in self.classes:
            rows = [X[i] for i in range(len(Y)) if Y[i] == c]

            self.class_priors[c] = len(rows) / n_total

            probs = []
            for f in range(n_features):
                count_ones = 0
                for row in rows:
                    if row[f] == 1:
                        count_ones += 1
                p = (count_ones + alpha) / (len(rows) + 2 * alpha)
                probs.append(p)

            self.feature_probs[c] = probs

    def predict_scores(self, X):
        all_scores = []
        for row in X:
            scores = {}
            for c in self.classes:
                score = log(self.class_priors[c])
                for f in range(len(row)):
                    p = self.feature_probs[c][f]
                    p = clip(p, 1e-9, 1 - 1e-9)
                    if row[f] == 1:
                        score += log(p)
                    else:
                        score += log(1 - p)
                scores[c] = score
            all_scores.append(scores)
        return all_scores

    def predict(self, X):
        scores = self.predict_scores(X)
        predictions = []
        for s in scores:
            best_class = None
            best_score = None
            for c in s:
                if best_score is None or s[c] > best_score:
                    best_score = s[c]
                    best_class = c
            predictions.append(best_class)
        return predictions

    def accuracy(self, X, Y):
        preds = self.predict(X)
        correct = 0
        for i in range(len(Y)):
            if preds[i] == Y[i]:
                correct += 1
        return correct / len(Y)


def generate_gaussian_data(n_per_class=100, seed=0):
    rng = Simple_LCG(seed)
    X = []
    Y = []

    for _ in range(n_per_class):
        x1 = 1.0 + rng.next_gaussian() * 0.6
        x2 = 1.0 + rng.next_gaussian() * 0.6
        X.append([x1, x2])
        Y.append(0)

    for _ in range(n_per_class):
        x1 = 4.0 + rng.next_gaussian() * 0.6
        x2 = 4.0 + rng.next_gaussian() * 0.6
        X.append([x1, x2])
        Y.append(1)

    for _ in range(n_per_class):
        x1 = 1.0 + rng.next_gaussian() * 0.6
        x2 = 4.0 + rng.next_gaussian() * 0.6
        X.append([x1, x2])
        Y.append(2)

    return X, Y


def generate_word_counts(n_per_class=80, n_features=10, seed=1):
    rng = Simple_LCG(seed)
    X = []
    Y = []

    for _ in range(n_per_class):
        row = []
        for f in range(n_features):
            if f < n_features // 2:
                count = int(rng.next_range(5, 20))
            else:
                count = int(rng.next_range(0, 4))
            row.append(count)
        X.append(row)
        Y.append(0)

    for _ in range(n_per_class):
        row = []
        for f in range(n_features):
            if f >= n_features // 2:
                count = int(rng.next_range(5, 20))
            else:
                count = int(rng.next_range(0, 4))
            row.append(count)
        X.append(row)
        Y.append(1)

    return X, Y


def generate_binary_docs(n_per_class=80, n_features=10, seed=2):
    rng = Simple_LCG(seed)
    X, Y = [], []

    for _ in range(n_per_class):
        row = []
        for f in range(n_features):
            if f < n_features // 2:
                val = 1 if rng.next_float(0, 1) > 0.25 else 0  # 75% chance of 1
            else:
                val = 1 if rng.next_float(0, 1) > 0.75 else 0  # 25% chance of 1
            row.append(val)
        X.append(row)
        Y.append(0)

    for _ in range(n_per_class):
        row = []
        for f in range(n_features):
            if f >= n_features // 2:
                val = 1 if rng.next_float(0, 1) > 0.25 else 0  # 75% chance of 1
            else:
                val = 1 if rng.next_float(0, 1) > 0.75 else 0  # 25% chance of 1
            row.append(val)
        X.append(row)
        Y.append(1)

    return X, Y

def per_class_accuracy(model, X_test, Y_test):
    preds = model.predict(X_test)
    classes = list(set(Y_test))
    classes.sort()
    for c in classes:
        total   = sum(1 for y in Y_test if y == c)
        correct = sum(1 for i in range(len(Y_test)) if Y_test[i] == c and preds[i] == c)
        if total > 0:
            print(f"    class {c}: {correct}/{total} = {correct/total:.2%}")


print("=" * 45)
print("TEST 1: GaussianNB on continuous blob data")
print("=" * 45)
X, Y = generate_gaussian_data(n_per_class=120)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
acc = gnb.accuracy(X_test, Y_test)
print(f"  Overall Accuracy: {acc:.2%}")
per_class_accuracy(gnb, X_test, Y_test)

print()
print("=" * 45)
print("TEST 2: MultinomialNB on word count data")
print("=" * 45)
X, Y = generate_word_counts(n_per_class=120)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
mnb = MultinomialNB()
mnb.fit(X_train, Y_train, alpha=1.0)
acc = mnb.accuracy(X_test, Y_test)
print(f"  Overall Accuracy: {acc:.2%}")
per_class_accuracy(mnb, X_test, Y_test)

print()
print("=" * 45)
print("TEST 3: BernoulliNB on binary document data")
print("=" * 45)
X, Y = generate_binary_docs(n_per_class=120)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
bnb = BernoulliNB()
bnb.fit(X_train, Y_train, alpha=1.0)
acc = bnb.accuracy(X_test, Y_test)
print(f"  Overall Accuracy: {acc:.2%}")
per_class_accuracy(bnb, X_test, Y_test)

print()
print("done.")
