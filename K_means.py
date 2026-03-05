from utilities import (
    mean, variance, std, unique, LCG,
    norm, vec_sub, dot,
    mean_squared_error, absolute,
)
from math import cos, sin

E  = 2.718281828459045
PI = 3.141592653589793

def euclidean(a, b):
    return norm(vec_sub(a, b), p=2)

def manhattan(a, b):
    return norm(vec_sub(a, b), p=1)

'''
 K-MEANS

Algorithm:
   1. Initialise K centroids
   2. Assign each point to nearest centroid
   3. Recompute centroids as mean of assigned points
   4. Repeat 2-3 until centroids stop moving (converged)

 Initialisation strategies:#   random    : pick K random points from data
   kmeans++  : spread initial centroids far apart
               — much better convergence in practice

'''

class KMeans:
    '''
    Parameters
    ----------
    k          : number of clusters
    max_iter   : maximum number of update iterations
    tol        : stop if centroid shift < tol
    init       : 'random' or 'kmeans++'
    n_init     : run full algorithm n_init times, keep best
    metric     : 'euclidean' or 'manhattan'
    seed       : random seed
    '''

    def __init__(self, k=3, max_iter=300, tol=1e-4,
                 init='kmeans++', n_init=10,
                 metric='euclidean', seed=42):
        self.k        = k
        self.max_iter = max_iter
        self.tol      = tol
        self.init     = init
        self.n_init   = n_init
        self.metric   = metric
        self.seed     = seed
        self.centroids   = None
        self.labels_     = None
        self.inertia_    = None
        self.n_iter_     = 0

        self._dist = euclidean if metric == 'euclidean' else manhattan


    def _init_random(self, X, rng):
        idx = rng.shuffle_indices(len(X))[:self.k]
        return [list(X[i]) for i in idx]

    def _init_kmeans_plus_plus(self, X, rng):
        n = len(X)
        first = rng.next_int(0, n - 1)
        centroids = [list(X[first])]

        for _ in range(self.k - 1):
            # squared distance from each point to nearest centroid
            dists = []
            for x in X:
                d = min(self._dist(x, c)**2 for c in centroids)
                dists.append(d)

            total = sum(dists)
            if total == 0:
                # all points coincide with existing centroids
                idx = rng.next_int(0, n - 1)
                centroids.append(list(X[idx]))
                continue

            probs      = [d / total for d in dists]
            cumulative = []
            running    = 0.0
            for p in probs:
                running += p
                cumulative.append(running)

            r = rng.next_float(0.0, 1.0)
            chosen = n - 1
            for i, cp in enumerate(cumulative):
                if r <= cp:
                    chosen = i
                    break
            centroids.append(list(X[chosen]))

        return centroids


    def _assign(self, X, centroids):
        labels = []
        for x in X:
            dists = [self._dist(x, c) for c in centroids]
            labels.append(dists.index(min(dists)))
        return labels


    def _update_centroids(self, X, labels):
        nf       = len(X[0])
        new_centroids = []
        for c in range(self.k):
            members = [X[i] for i in range(len(X)) if labels[i] == c]
            if not members:
                new_centroids.append(self.centroids[c])
            else:
                new_c = [mean([m[j] for m in members]) for j in range(nf)]
                new_centroids.append(new_c)
        return new_centroids


    def _shift(self, old, new):
        '''Max centroid movement across all centroids.'''
        return max(self._dist(old[c], new[c]) for c in range(self.k))


    def _inertia(self, X, labels, centroids):
        '''
        WCSS = sum of squared distances from each point to its centroid.
        Lower is better. Used to compare runs and evaluate k.
        '''
        return sum(self._dist(X[i], centroids[labels[i]])**2
                   for i in range(len(X)))


    def _fit_once(self, X, rng):
        if self.init == 'kmeans++':
            centroids = self._init_kmeans_plus_plus(X, rng)
        else:
            centroids = self._init_random(X, rng)

        labels = self._assign(X, centroids)

        for it in range(self.max_iter):
            new_centroids = self._update_centroids(X, labels)
            shift         = self._shift(centroids, new_centroids)
            centroids     = new_centroids
            labels        = self._assign(X, centroids)
            if shift < self.tol:
                return centroids, labels, it + 1, self._inertia(X, labels, centroids)

        return centroids, labels, self.max_iter, self._inertia(X, labels, centroids)


    def fit(self, X):
        best_inertia   = float('inf')
        best_centroids = None
        best_labels    = None
        best_iters     = 0

        for run in range(self.n_init):
            rng = LCG(self.seed + run * 1000)
            centroids, labels, iters, inertia = self._fit_once(X, rng)
            if inertia < best_inertia:
                best_inertia   = inertia
                best_centroids = centroids
                best_labels    = labels
                best_iters     = iters

        self.centroids = best_centroids
        self.labels_   = best_labels
        self.inertia_  = best_inertia
        self.n_iter_   = best_iters
        return self

    def predict(self, X):
        return self._assign(X, self.centroids)

    def fit_predict(self, X):
        return self.fit(X).labels_


    def score_report(self, X, name=''):
        tag = f' [{name}]' if name else ''
        print(f'  Inertia{tag}    : {self.inertia_:.4f}')
        print(f'  Iterations     : {self.n_iter_}')
        print(f'  Cluster sizes  :')
        for c in range(self.k):
            size = sum(1 for l in self.labels_ if l == c)
            bar  = '&' * int(30 * size / len(X))
            print(f'    cluster {c}: {size:>4} pts  {bar}')


def elbow(X, k_range=None, metric='euclidean', seed=42):
    if k_range is None:
        k_range = range(1, 11)
    results = []
    for k in k_range:
        km = KMeans(k=k, metric=metric, seed=seed, n_init=5)
        km.fit(X)
        results.append((k, km.inertia_))
        print(f'  k={k:>2}  inertia={km.inertia_:>10.2f}')
    return results

def silhouette_score(X, labels, metric='euclidean'):
    dist = euclidean if metric == 'euclidean' else manhattan
    n     = len(X)
    k     = max(labels) + 1
    scores = []

    for i in range(n):
        ci = labels[i]

        # a(i): mean dist to same-cluster points
        same = [j for j in range(n) if labels[j] == ci and j != i]
        if not same:
            scores.append(0.0)
            continue
        a = mean([dist(X[i], X[j]) for j in same])

        # b(i): mean dist to nearest other cluster
        b = float('inf')
        for c in range(k):
            if c == ci:
                continue
            other = [j for j in range(n) if labels[j] == c]
            if not other:
                continue
            b_c = mean([dist(X[i], X[j]) for j in other])
            if b_c < b:
                b = b_c

        s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        scores.append(s)

    return mean(scores)


class MiniBatchKMeans:
    def __init__(self, k=3, n_iter=100, batch_size=100,
                 seed=42, tol=1e-4):
        self.k          = k
        self.n_iter     = n_iter
        self.batch_size = batch_size
        self.seed       = seed
        self.tol        = tol
        self.centroids  = None
        self.labels_    = None

    def fit(self, X):
        n   = len(X)
        rng = LCG(self.seed)
        nf  = len(X[0])

        subset_idx = rng.shuffle_indices(n)[:min(500, n)]
        Xs = [X[i] for i in subset_idx]
        km_init = KMeans(k=self.k, n_init=1, seed=self.seed)
        km_init._fit_once(Xs, rng)
        centroids = km_init._init_kmeans_plus_plus(Xs, rng)

        counts = [0] * self.k

        for it in range(self.n_iter):
            # sample a mini-batch
            batch_idx = [rng.next_int(0, n-1) for _ in range(self.batch_size)]
            batch     = [X[i] for i in batch_idx]
            batch_labels = []
            for x in batch:
                dists = [euclidean(x, c) for c in centroids]
                batch_labels.append(dists.index(min(dists)))
            old = [list(c) for c in centroids]
            for x, c in zip(batch, batch_labels):
                counts[c] += 1
                lr = 1.0 / counts[c]
                centroids[c] = [centroids[c][j] + lr * (x[j] - centroids[c][j])for j in range(nf)]

            shift = max(euclidean(old[c], centroids[c]) for c in range(self.k))
            if shift < self.tol:
                break

        self.centroids = centroids
        self.labels_   = self._assign(X)
        return self

    def _assign(self, X):
        return [min(range(self.k), key=lambda c: euclidean(x, self.centroids[c]))
                for x in X]

    def predict(self, X):
        return self._assign(X)


def cos_a(x):
  return cos(x)

def sin_a(x):
  return sin(x)
  
def gen_blobs(n=300, n_clusters=3, spread=0.5, seed=1):
    g = LCG(seed)
    centers = [[0,3],[-3,-1.5],[3,-1.5],[-3,3],[3,3],
               [0,-3],[0,0],[-1,2],[2,1],[-2,-2]][:n_clusters]
    X, true_labels = [], []
    per = n // n_clusters
    for c, ce in enumerate(centers):
        for _ in range(per):
            X.append([ce[0]+g.next_gaussian()*spread,
                      ce[1]+g.next_gaussian()*spread])
            true_labels.append(c)
    return X, true_labels

def gen_circles(n=300, seed=3):
    g = LCG(seed); X, Y = [], []
    for i in range(n):
        a = g.next_float(0, 2*PI)
        if i % 3 == 0:   r = 0.4 + g.next_gaussian()*0.05
        elif i % 3 == 1: r = 1.0 + g.next_gaussian()*0.05
        else:             r = 1.8 + g.next_gaussian()*0.05
        X.append([r*cos_a(a), r*sin_a(a)])
        Y.append(i % 3)
    return X, Y

def purity_score(true_labels, pred_labels):
    n = len(true_labels)
    k = max(pred_labels) + 1
    total = 0
    for c in range(k):
        members = [true_labels[i] for i in range(n) if pred_labels[i] == c]
        if members:
            counts  = {}
            for m in members: counts[m] = counts.get(m,0)+1
            total  += max(counts.values())
    return total / n

sep = '='*52

print(sep)
print('TEST 1 — K-Means on Gaussian Blobs (k=3)')
print(sep)
X, true = gen_blobs(300, 3)
km = KMeans(k=3, init='kmeans++', n_init=10, seed=42)
km.fit(X)
print()
km.score_report(X, name='kmeans++')
purity = purity_score(true, km.labels_)
sil    = silhouette_score(X, km.labels_)
print(f'  Purity         : {purity:.4f}')
print(f'  Silhouette     : {sil:.4f}')

print()
print(sep)
print('TEST 2 — Elbow Method (finding best k)')
print(sep)
print()
elbow_res = elbow(X, k_range=range(1, 8), seed=42)

print()
print(sep)
print('TEST 3 — KMeans++ vs Random Init')
print(sep)
print()
for init in ['random', 'kmeans++']:
    inertias = []
    for seed in range(10):
        km_i = KMeans(k=3, init=init, n_init=1, seed=seed)
        km_i.fit(X)
        inertias.append(km_i.inertia_)
    print(f'  {init:<12}  mean_inertia={mean(inertias):.2f}'
          f'  std={std(inertias):.2f}'
          f'  min={min(inertias):.2f}  max={max(inertias):.2f}')

print()
print(sep)
print('TEST 4 — Silhouette score for different k values')
print(sep)
print()
for k in range(2, 8):
    km_k = KMeans(k=k, n_init=5, seed=42)
    km_k.fit(X)
    sil = silhouette_score(X, km_k.labels_)
    bar = '-' * int(sil * 30)
    print(f'  k={k}  silhouette={sil:.4f}  {bar}')

print()
print(sep)
print('TEST 5 — Concentric Circles (k-means struggles here)')
print('  K-Means assumes convex clusters — circles violate this')
print(sep)
Xc, true_c = gen_circles(300)
km_c = KMeans(k=3, n_init=10, seed=42)
km_c.fit(Xc)
print()
km_c.score_report(Xc, name='circles')
purity_c = purity_score(true_c, km_c.labels_)
sil_c    = silhouette_score(Xc, km_c.labels_)
print(f'  Purity         : {purity_c:.4f}  (1.0 = perfect)')
print(f'  Silhouette     : {sil_c:.4f}  (low = overlapping clusters)')

print()
print(sep)
print('TEST 6 — Mini-Batch K-Means vs Full K-Means')
print(sep)
X_large, true_large = gen_blobs(1000, 4, spread=0.6, seed=9)
print()

km_full = KMeans(k=4, n_init=5, seed=42)
km_full.fit(X_large)
sil_full = silhouette_score(X_large, km_full.labels_)
print(f'  Full KMeans    inertia={km_full.inertia_:.2f}  '
      f'silhouette={sil_full:.4f}  iters={km_full.n_iter_}')

km_mini = MiniBatchKMeans(k=4, n_iter=100, batch_size=100, seed=42)
km_mini.fit(X_large)
mini_inertia = sum(euclidean(X_large[i], km_mini.centroids[km_mini.labels_[i]])**2
                   for i in range(len(X_large)))
sil_mini = silhouette_score(X_large, km_mini.labels_)
print(f'  MiniBatch KM   inertia={mini_inertia:.2f}  '
      f'silhouette={sil_mini:.4f}')

print()
print(sep)
print('TEST 7 — Manhattan vs Euclidean K-Means')
print(sep)
print()
for metric in ['euclidean', 'manhattan']:
    km_m = KMeans(k=3, metric=metric, n_init=5, seed=42)
    km_m.fit(X)
    sil = silhouette_score(X, km_m.labels_)
    print(f'  {metric:<12}  inertia={km_m.inertia_:.2f}  silhouette={sil:.4f}')
