from utilities import mean, LCG, accuracy, log, exp, sqrt, absolute

E = 2.718281828459045

'''
CNNs are layered structured typically using:
  Input image (H x W x C)
  Conv layer (filters, relu)
  MaxPool
  Flatten
  Dense (relu)
  Output (sigmoid / softmax)
I will 8*8 or 16*16 images for fast compute. Also not exploring the diversities of the layers.
'''




def make_zeros(shape):
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [make_zeros(shape[1:]) for _ in range(shape[0])]

def randn(rng):
    return rng.next_gaussian()

def relu(x):
    return max(0.0, x)

def relu_d(x):
    return 1.0 if x > 0 else 0.0

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + E ** (-x))

def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(scores):
    m = max(scores)
    e = [E ** (s - m) for s in scores]
    t = sum(e)
    return [ei / t for ei in e]

'''
Let's create the backbone of CNNs: Convolution layer

filter shape : (n_filters, filter_H, filter_W, n_channels)
input  shape : (H, W, C)
output shape : (H - fH + 1, W - fW + 1, n_filters)   (no padding)

forward:
  for each filter f:
    for each position (i, j):
      out[i][j][f] = sum over (fh, fw, c) of
                     input[i+fh][j+fw][c] * filter[f][fh][fw][c]
                     + bias[f]
      out[i][j][f] = relu(out[i][j][f])
'''


class ConvLayer:

    def __init__(self, n_filters, filter_size, n_channels, seed=0):
        self.n_filters   = n_filters
        self.filter_size = filter_size
        self.n_channels  = n_channels

        rng   = LCG(seed)
        scale = sqrt(2.0 / (filter_size * filter_size * n_channels))

        self.filters = []
        for f in range(n_filters):
            filt = []
            for fh in range(filter_size):
                row = []
                for fw in range(filter_size):
                    ch = []
                    for c in range(n_channels):
                        ch.append(randn(rng) * scale)
                    row.append(ch)
                filt.append(row)
            self.filters.append(filt)

        self.biases = [0.0] * n_filters

        self.input   = None
        self.pre_act = None
        self.output  = None

    def forward(self, inp):
        H  = len(inp)
        W  = len(inp[0])
        fS = self.filter_size
        oH = H - fS + 1
        oW = W - fS + 1

        self.input = inp

        pre_act = []
        for i in range(oH):
            row = []
            for j in range(oW):
                pixel = []
                for f in range(self.n_filters):
                    val = self.biases[f]
                    for fh in range(fS):
                        for fw in range(fS):
                            for c in range(self.n_channels):
                                val += inp[i+fh][j+fw][c] * self.filters[f][fh][fw][c]
                    pixel.append(val)
                row.append(pixel)
            pre_act.append(row)

        self.pre_act = pre_act

        out = []
        for i in range(oH):
            row = []
            for j in range(oW):
                pixel = []
                for f in range(self.n_filters):
                    pixel.append(relu(pre_act[i][j][f]))
                row.append(pixel)
            out.append(row)

        self.output = out
        return out

    def backward(self, d_out, lr):
        H  = len(self.input)
        W  = len(self.input[0])
        fS = self.filter_size
        oH = len(d_out)
        oW = len(d_out[0])

        d_filters = []
        for f in range(self.n_filters):
            df = []
            for fh in range(fS):
                row = []
                for fw in range(fS):
                    ch = [0.0] * self.n_channels
                    row.append(ch)
                df.append(row)
            d_filters.append(df)

        d_biases = [0.0] * self.n_filters

        d_inp = []
        for i in range(H):
            row = []
            for j in range(W):
                row.append([0.0] * self.n_channels)
            d_inp.append(row)

        for i in range(oH):
            for j in range(oW):
                for f in range(self.n_filters):
                    grad = d_out[i][j][f] * relu_d(self.pre_act[i][j][f])
                    d_biases[f] += grad
                    for fh in range(fS):
                        for fw in range(fS):
                            for c in range(self.n_channels):
                                d_filters[f][fh][fw][c] += grad * self.input[i+fh][j+fw][c]
                                d_inp[i+fh][j+fw][c]    += grad * self.filters[f][fh][fw][c]

        for f in range(self.n_filters):
            self.biases[f] -= lr * d_biases[f]
            for fh in range(fS):
                for fw in range(fS):
                    for c in range(self.n_channels):
                        self.filters[f][fh][fw][c] -= lr * d_filters[f][fh][fw][c]

        return d_inp


class MaxPoolLayer:

    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.input     = None
        self.max_mask  = None

    def forward(self, inp):
        H  = len(inp)
        W  = len(inp[0])
        C  = len(inp[0][0])
        p  = self.pool_size
        oH = H // p
        oW = W // p

        self.input = inp

        mask = []
        for i in range(H):
            row = []
            for j in range(W):
                row.append([False] * C)
            mask.append(row)

        out = []
        for i in range(oH):
            row = []
            for j in range(oW):
                pixel = []
                for c in range(C):
                    max_val = inp[i*p][j*p][c]
                    max_i, max_j = i*p, j*p
                    for pi in range(p):
                        for pj in range(p):
                            v = inp[i*p+pi][j*p+pj][c]
                            if v > max_val:
                                max_val  = v
                                max_i, max_j = i*p+pi, j*p+pj
                    pixel.append(max_val)
                    mask[max_i][max_j][c] = True
                row.append(pixel)
            out.append(row)

        self.max_mask = mask
        return out

    def backward(self, d_out):
        H  = len(self.input)
        W  = len(self.input[0])
        C  = len(self.input[0][0])
        p  = self.pool_size
        oH = len(d_out)
        oW = len(d_out[0])

        d_inp = []
        for i in range(H):
            row = []
            for j in range(W):
                row.append([0.0] * C)
            d_inp.append(row)

        for i in range(oH):
            for j in range(oW):
                for c in range(C):
                    for pi in range(p):
                        for pj in range(p):
                            if self.max_mask[i*p+pi][j*p+pj][c]:
                                d_inp[i*p+pi][j*p+pj][c] = d_out[i][j][c]

        return d_inp


def flatten(inp):
    flat = []
    for row in inp:
        for pixel in row:
            for v in pixel:
                flat.append(v)
    return flat

def unflatten(flat, H, W, C):
    out = []
    idx = 0
    for i in range(H):
        row = []
        for j in range(W):
            pixel = []
            for c in range(C):
                pixel.append(flat[idx])
                idx += 1
            row.append(pixel)
        out.append(row)
    return out


class DenseLayer:

    def __init__(self, n_in, n_out, activation='relu', seed=0):
        rng   = LCG(seed)
        scale = sqrt(2.0 / n_in) if activation == 'relu' else sqrt(1.0 / n_in)

        self.W          = [[randn(rng) * scale for _ in range(n_in)] for _ in range(n_out)]
        self.b          = [0.0] * n_out
        self.activation = activation
        self.inp        = None
        self.z          = None
        self.out        = None

    def forward(self, inp):
        self.inp = inp
        self.z   = []
        for j in range(len(self.W)):
            val = self.b[j]
            for k in range(len(inp)):
                val += self.W[j][k] * inp[k]
            self.z.append(val)

        if self.activation == 'relu':
            self.out = [relu(zi) for zi in self.z]
        elif self.activation == 'sigmoid':
            self.out = [sigmoid(zi) for zi in self.z]
        elif self.activation == 'softmax':
            self.out = softmax(self.z)
        else:
            self.out = list(self.z)

        return self.out

    def backward(self, delta, lr):
        n_in  = len(self.inp)
        n_out = len(self.W)

        if self.activation == 'relu':
            delta_act = [delta[j] * relu_d(self.z[j]) for j in range(n_out)]
        elif self.activation == 'sigmoid':
            delta_act = [delta[j] * sigmoid_d(self.z[j]) for j in range(n_out)]
        else:
            delta_act = list(delta)

        d_inp = []
        for k in range(n_in):
            val = 0.0
            for j in range(n_out):
                val += self.W[j][k] * delta_act[j]
            d_inp.append(val)

        for j in range(n_out):
            self.b[j] -= lr * delta_act[j]
            for k in range(n_in):
                self.W[j][k] -= lr * delta_act[j] * self.inp[k]

        return d_inp


class CNN:

    def __init__(self, n_filters=4, filter_size=3,
                 dense_size=32, n_classes=2,
                 lr=0.01, epochs=10, seed=42):
        self.n_filters   = n_filters
        self.filter_size = filter_size
        self.dense_size  = dense_size
        self.n_classes   = n_classes
        self.lr          = lr
        self.epochs      = epochs
        self.seed        = seed
        self.loss_history = []

    def _build(self, img_H, img_W, img_C):
        self.conv   = ConvLayer(self.n_filters, self.filter_size,
                                img_C, seed=self.seed)
        self.pool   = MaxPoolLayer(pool_size=2)

        oH = (img_H - self.filter_size + 1) // 2
        oW = (img_W - self.filter_size + 1) // 2
        flat_size = oH * oW * self.n_filters

        self.pool_oH = oH
        self.pool_oW = oW

        self.dense1 = DenseLayer(flat_size, self.dense_size,
                                 activation='relu', seed=self.seed+1)

        out_act = 'sigmoid' if self.n_classes == 2 else 'softmax'
        n_out   = 1 if self.n_classes == 2 else self.n_classes
        self.dense2 = DenseLayer(self.dense_size, n_out,
                                 activation=out_act, seed=self.seed+2)

    def _forward(self, img):
        x = self.conv.forward(img)
        x = self.pool.forward(x)
        x = flatten(x)
        x = self.dense1.forward(x)
        x = self.dense2.forward(x)
        return x

    def _loss_and_delta(self, pred, y_true):
        if self.n_classes == 2:
            p     = max(1e-12, min(1-1e-12, pred[0]))
            loss  = -(y_true * log(p) + (1 - y_true) * log(1 - p))
            delta = [pred[0] - y_true]
        else:
            loss  = -sum(y_true[k] * log(max(1e-12, pred[k]))
                         for k in range(self.n_classes))
            delta = [pred[k] - y_true[k] for k in range(self.n_classes)]
        return loss, delta

    def _backward(self, delta):
        d = self.dense2.backward(delta, self.lr)
        d = self.dense1.backward(d, self.lr)
        d = unflatten(d, self.pool_oH, self.pool_oW, self.n_filters)
        d = self.pool.backward(d)
        self.conv.backward(d, self.lr)

    def fit(self, X, Y):
        img_H = len(X[0])
        img_W = len(X[0][0])
        img_C = len(X[0][0][0])
        self._build(img_H, img_W, img_C)

        if self.n_classes == 2:
            Yn = [float(y) for y in Y]
        else:
            Yn = [[1.0 if y == k else 0.0 for k in range(self.n_classes)]
                  for y in Y]

        rng = LCG(self.seed)

        for epoch in range(self.epochs):
            indices   = rng.shuffle_indices(len(X))
            epoch_loss = 0.0
            for i in indices:
                pred = self._forward(X[i])
                loss, delta = self._loss_and_delta(pred, Yn[i])
                epoch_loss += loss
                self._backward(delta)
            avg_loss = epoch_loss / len(X)
            self.loss_history.append(avg_loss)
            print(f'  epoch {epoch+1:>3}/{self.epochs}  loss={avg_loss:.4f}')

        return self

    def predict(self, X):
        preds = []
        for img in X:
            out = self._forward(img)
            if self.n_classes == 2:
                preds.append(1 if out[0] >= 0.5 else 0)
            else:
                preds.append(out.index(max(out)))
        return preds

    def accuracy(self, X, Y):
        return accuracy(Y, self.predict(X))


def make_image(H, W, C, rng):
    img = []
    for i in range(H):
        row = []
        for j in range(W):
            pixel = [rng.next_float(0.0, 1.0) for _ in range(C)]
            row.append(pixel)
        img.append(row)
    return img

def gen_horizontal_vs_vertical(n=100, img_size=8, seed=1):
    '''
    Class 0: image with a bright horizontal stripe
    Class 1: image with a bright vertical stripe
    CNN should learn edge filters to distinguish them.
    '''
    rng = LCG(seed)
    X, Y = [], []
    for i in range(n):
        img = make_image(img_size, img_size, 1, rng)
        if i % 2 == 0:
            stripe = rng.next_int(1, img_size-2)
            for j in range(img_size):
                img[stripe][j][0] = 1.0
            Y.append(0)
        else:
            stripe = rng.next_int(1, img_size-2)
            for j in range(img_size):
                img[j][stripe][0] = 1.0
            Y.append(1)
        X.append(img)
    return X, Y

def gen_bright_vs_dark(n=100, img_size=8, seed=2):
    rng = LCG(seed)
    X, Y = [], []
    for i in range(n):
        if i % 2 == 0:
            img = make_image(img_size, img_size, 1, rng)
            for r in range(img_size):
                for c in range(img_size):
                    img[r][c][0] = rng.next_float(0.0, 0.4)
            Y.append(0)
        else:
            img = make_image(img_size, img_size, 1, rng)
            for r in range(img_size):
                for c in range(img_size):
                    img[r][c][0] = rng.next_float(0.6, 1.0)
            Y.append(1)
        X.append(img)
    return X, Y

def gen_quadrant(n=200, img_size=8, seed=3):
    rng = LCG(seed)
    X, Y = [], []
    half = img_size // 2
    for i in range(n):
        img = make_image(img_size, img_size, 1, rng)
        for r in range(img_size):
            for c in range(img_size):
                img[r][c][0] = 0.1
        label  = i % 4
        r_start = 0 if label in [0, 1] else half
        c_start = 0 if label in [0, 2] else half
        for r in range(r_start, r_start + half):
            for c in range(c_start, c_start + half):
                img[r][c][0] = 0.9
        X.append(img)
        Y.append(label)
    return X, Y

sep = '='*52

print(sep)
print('TEST 1 — CNN: bright vs dark images (binary)')
print(sep)
X, Y = gen_bright_vs_dark(n=120, img_size=8)
split = int(len(X) * 0.8)
Xtr, Xte = X[:split], X[split:]
Ytr, Yte = Y[:split], Y[split:]
print()
cnn1 = CNN(n_filters=4, filter_size=3, dense_size=16,
           n_classes=2, lr=0.05, epochs=10, seed=42)
cnn1.fit(Xtr, Ytr)
print()
acc = cnn1.accuracy(Xte, Yte)
print(f'  Test accuracy: {acc*100:.2f}%')

print()
print(sep)
print('TEST 2 — CNN: horizontal vs vertical stripe')
print(sep)
X2, Y2 = gen_horizontal_vs_vertical(n=120, img_size=8)
split2  = int(len(X2) * 0.8)
Xtr2, Xte2 = X2[:split2], X2[split2:]
Ytr2, Yte2 = Y2[:split2], Y2[split2:]
print()
cnn2 = CNN(n_filters=8, filter_size=3, dense_size=16,
           n_classes=2, lr=0.05, epochs=15, seed=42)
cnn2.fit(Xtr2, Ytr2)
print()
acc2 = cnn2.accuracy(Xte2, Yte2)
print(f'  Test accuracy: {acc2*100:.2f}%')

print()
print(sep)
print('TEST 3 — CNN: 4-class quadrant classification')
print(sep)
X3, Y3 = gen_quadrant(n=200, img_size=8)
split3  = int(len(X3) * 0.8)
Xtr3, Xte3 = X3[:split3], X3[split3:]
Ytr3, Yte3 = Y3[:split3], Y3[split3:]
print()
cnn3 = CNN(n_filters=8, filter_size=3, dense_size=32,
           n_classes=4, lr=0.02, epochs=20, seed=42)
cnn3.fit(Xtr3, Ytr3)
print()
acc3 = cnn3.accuracy(Xte3, Yte3)
print(f'  Test accuracy: {acc3*100:.2f}%')

print()
print(sep)
print('TEST 4 — What the filters learned')
print('  (showing first conv filter weights)')
print(sep)
print()
for f in range(cnn2.n_filters):
    print(f'  Filter {f}:')
    for row in cnn2.conv.filters[f]:
        vals = [round(row[col][0], 3) for col in range(len(row))]
        print(f'    {vals}')
