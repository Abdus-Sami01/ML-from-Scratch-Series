from utilities import (
    mean, std, unique, LCG,
    norm, vec_sub, absolute,
    gen_blobs, gen_circles,
    PI, cos, sin, manhattan, eucleidian
)

NOISE     = -1
UNVISITED = -2
METRICS = {'euclidean': euclidean, 'manhattan': manhattan}


class DBSCAN:
    '''
    Parameters
    ----------
    eps      : neighbourhood radius
    min_pts  : minimum points to be a core point
    metric   : 'euclidean' or 'manhattan'
    '''

    def __init__(self, eps=0.5, min_pts=5, metric='euclidean'):
        self.eps     = eps
        self.min_pts = min_pts
        self.dist    = METRICS[metric]
        self.labels_ = None
        self.core_mask_ = None

    def _region_query(self, X, i):
        '''Return indices of all points within eps of X[i].'''
        return [j for j in range(len(X))
                if self.dist(X[i], X[j]) <= self.eps]

    def fit(self, X):
        n            = len(X)
        labels       = [UNVISITED] * n
        cluster_id   = 0

        for i in range(n):
            if labels[i] != UNVISITED:
                continue

            neighbours = self._region_query(X, i)

            if len(neighbours) < self.min_pts:
                labels[i] = NOISE
                continue

            # start new cluster from core point i
            labels[i] = cluster_id
            seeds      = list(neighbours)
            seeds.remove(i)

            si = 0
            while si < len(seeds):
                q = seeds[si]
                if labels[q] == NOISE:
                    labels[q] = cluster_id
                if labels[q] != UNVISITED:
                    si += 1
                    continue

                labels[q]    = cluster_id
                q_neighbours = self._region_query(X, q)

                if len(q_neighbours) >= self.min_pts:
                    for nb in q_neighbours:
                        if nb not in seeds:
                            seeds.append(nb)
                si += 1

            cluster_id += 1

        self.labels_     = labels
        self.core_mask_  = self._find_cores(X)
        self.n_clusters_ = cluster_id
        self.n_noise_    = labels.count(NOISE)
        return self

    def _find_cores(self, X):
        return [len(self._region_query(X, i)) >= self.min_pts
                for i in range(len(X))]

    def fit_predict(self, X):
        return self.fit(X).labels_

    def score_report(self, X, name=''):
        tag = f' [{name}]' if name else ''
        print(f'  Clusters{tag}  : {self.n_clusters_}')
        print(f'  Noise pts  : {self.n_noise_} / {len(X)}'
              f' ({self.n_noise_/len(X)*100:.1f}%)')
        for c in range(self.n_clusters_):
            size = sum(1 for l in self.labels_ if l == c)
            bar  = '&' * int(30 * size / len(X))
            print(f'    cluster {c}: {size:>4} pts  {bar}')


def purity_score(true_labels, pred_labels):
    n = len(true_labels)
    clusters = unique(pred_labels)
    total = 0
    for c in clusters:
        if c == NOISE:
            continue
        members = [true_labels[i] for i in range(n) if pred_labels[i] == c]
        if members:
            counts = {}
            for m in members: counts[m] = counts.get(m, 0) + 1
            total += max(counts.values())
    return total / n

def silhouette_score(X, labels, metric='euclidean'):
    dist   = METRICS[metric]
    n      = len(X)
    non_noise = [i for i in range(n) if labels[i] != NOISE]
    if len(non_noise) < 2:
        return 0.0
    scores = []
    for i in non_noise:
        ci   = labels[i]
        same = [j for j in non_noise if labels[j] == ci and j != i]
        if not same:
            scores.append(0.0)
            continue
        a = mean([dist(X[i], X[j]) for j in same])
        b = float('inf')
        for c in unique([labels[j] for j in non_noise]):
            if c == ci or c == NOISE: continue
            other = [j for j in non_noise if labels[j] == c]
            if not other: continue
            b_c = mean([dist(X[i], X[j]) for j in other])
            if b_c < b: b = b_c
        if b == float('inf'):
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return mean(scores) if scores else 0.0


def k_distance(X, k=4, metric='euclidean'):
    dist  = METRICS[metric]
    n     = len(X)
    kdists = []
    for i in range(n):
        dists = sorted([dist(X[i], X[j]) for j in range(n) if j != i])
        kdists.append(dists[k-1])
    kdists.sort()
    return kdists

def eps_suggestion(X, k=4, metric='euclidean'):
    kdists = k_distance(X, k, metric)
    n      = len(kdists)
    diffs  = [kdists[i+1] - kdists[i] for i in range(n-1)]
    elbow  = diffs.index(max(diffs))
    suggested = kdists[elbow]
    print(f'  k-distance stats (k={k}):')
    print(f'    min    : {min(kdists):.4f}')
    print(f'    median : {kdists[n//2]:.4f}')
    print(f'    max    : {max(kdists):.4f}')
    print(f'    suggested eps (elbow) : {suggested:.4f}')
    return suggested


def gen_moons(n=300, seed=3):
    rng = LCG(seed); X, Y = [], []
    half = n // 2
    for _ in range(half):
        a = rng.next_float(0, PI)
        X.append([cos(a) + rng.next_gaussian()*0.1,
                  sin(a) + rng.next_gaussian()*0.1])
        Y.append(0)
    for _ in range(half):
        a = rng.next_float(0, PI)
        X.append([1 - cos(a) + rng.next_gaussian()*0.1,
                  0.5 - sin(a) + rng.next_gaussian()*0.1])
        Y.append(1)
    return X, Y

def gen_noise(n=300, n_clusters=3, seed=42):
    X, Y = gen_blobs(n, n_clusters, spread=0.4, seed=seed)
    rng  = LCG(seed + 99)
    for _ in range(n // 10):
        X.append([rng.next_float(-5, 5), rng.next_float(-5, 5)])
        Y.append(-1)
    return X, Y


sep = '='*52

print(sep)
print('TEST 1 — DBSCAN on Gaussian Blobs')
print(sep)
X, true = gen_blobs(300, 3, spread=0.5)
db = DBSCAN(eps=0.6, min_pts=5)
db.fit(X)
print()
db.score_report(X, name='blobs')
print(f'  Purity     : {purity_score(true, db.labels_):.4f}')
print(f'  Silhouette : {silhouette_score(X, db.labels_):.4f}')

print()
print(sep)
print('TEST 2 — DBSCAN on Concentric Circles')
print('  K-Means fails here. DBSCAN should succeed.')
print(sep)
Xc, true_c = gen_circles(600, n_rings=3)
db2 = DBSCAN(eps=0.15, min_pts=4)
db2.fit(Xc)
print()
db2.score_report(Xc, name='circles')
print(f'  Purity     : {purity_score(true_c, db2.labels_):.4f}')
print(f'  Silhouette : {silhouette_score(Xc, db2.labels_):.4f}')

print()
print(sep)
print('TEST 3 — DBSCAN on Interleaved Moons')
print(sep)
Xm, true_m = gen_moons(300)
db3 = DBSCAN(eps=0.2, min_pts=5)
db3.fit(Xm)
print()
db3.score_report(Xm, name='moons')
print(f'  Purity     : {purity_score(true_m, db3.labels_):.4f}')

print()
print(sep)
print('TEST 4 — DBSCAN with Noise Points')
print('  Points far from any cluster get label -1')
print(sep)
Xn, true_n = gen_noise(300, 3)
db4 = DBSCAN(eps=0.5, min_pts=5)
db4.fit(Xn)
print()
db4.score_report(Xn, name='noisy')

print()
print(sep)
print('TEST 5 — eps sensitivity')
print(sep)
X5, _ = gen_blobs(200, 3, spread=0.5)
print()
for eps in [0.2, 0.4, 0.6, 0.8, 1.2, 2.0]:
    db_e = DBSCAN(eps=eps, min_pts=5)
    db_e.fit(X5)
    print(f'  eps={eps:<4}  clusters={db_e.n_clusters_}  '
          f'noise={db_e.n_noise_:>3}  '
          f'({db_e.n_noise_/len(X5)*100:.0f}%)')

print()
print(sep)
print('TEST 6 — min_pts sensitivity')
print(sep)
print()
for mp in [2, 3, 5, 10, 15, 20]:
    db_m = DBSCAN(eps=0.6, min_pts=mp)
    db_m.fit(X5)
    print(f'  min_pts={mp:<3}  clusters={db_m.n_clusters_}  '
          f'noise={db_m.n_noise_:>3}')

print()
print(sep)
print('TEST 7 — eps suggestion via k-distance')
print(sep)
X7, _ = gen_blobs(200, 3, spread=0.5)
print()
suggested = eps_suggestion(X7, k=5)
db7 = DBSCAN(eps=suggested, min_pts=5)
db7.fit(X7)
print(f'  With suggested eps={suggested:.4f}:')
print(f'    clusters={db7.n_clusters_}  noise={db7.n_noise_}')
