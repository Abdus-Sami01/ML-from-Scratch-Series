from utilities import (
    mean, LCG, log, exp, sqrt, absolute,
    sigmoid, softmax, sin, cos, PI,
    accuracy, mean_squared_error,
    train_test_split,
)

E = 2.718281828459045

'''
A recurrent neural network processes sequences. 
At each time step t it takes: x_t (current input) and h_t-1 (hidden state from previous step). 
It produces: h_t = tanh(Wx * x_t + Wh * h_t-1 + b_h) and y_t = Wy * h_t + b_y. 
The hidden state h is the 'memory' of the network. 
It carries information from previous time steps forward. 
Training: Backpropagation Through Time (BPTT). Unroll the RNN for T steps. 
Compute loss at the end (or at each step). 
Backprop through all T steps in reverse. Gradients from all steps accumulate into Wx, Wh, b_h. 
Vanishing gradient problem: As T grows, gradients shrink exponentially. 
Solution: LSTM (gated memory) — built later in this file.
'''

def tanh(x):
    if x > 300:  return 1.0
    if x < -300: return -1.0
    ep = E ** x
    en = E ** (-x)
    return (ep - en) / (ep + en)

def tanh_d(x):
    t = tanh(x)
    return 1.0 - t * t

def clip(x, lo, hi):
    return max(lo, min(hi, x))


class RNN:
   
    def __init__(self, input_size, hidden_size, output_size,
                 lr=0.01, clip_val=5.0, seed=42):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr          = lr
        self.clip_val    = clip_val

        rng = LCG(seed)
        s   = sqrt(1.0 / hidden_size)

        self.Wx = [[rng.next_gaussian() * s for _ in range(input_size)]
                   for _ in range(hidden_size)]
        self.Wh = [[rng.next_gaussian() * s for _ in range(hidden_size)]
                   for _ in range(hidden_size)]
        self.bh = [0.0] * hidden_size

        self.Wy = [[rng.next_gaussian() * s for _ in range(hidden_size)]
                   for _ in range(output_size)]
        self.by = [0.0] * output_size

    def _mat_vec(self, M, v):
        out = []
        for row in M:
            val = 0.0
            for i in range(len(v)):
                val += row[i] * v[i]
            out.append(val)
        return out

    def _vec_add(self, a, b):
        return [a[i] + b[i] for i in range(len(a))]

    def forward(self, sequence):
        
        h = [0.0] * self.hidden_size
        cache = []

        for x in sequence:
            z = self._vec_add(
                self._vec_add(self._mat_vec(self.Wx, x),
                              self._mat_vec(self.Wh, h)),
                self.bh
            )
            h_new = [tanh(zi) for zi in z]
            y_raw = self._vec_add(self._mat_vec(self.Wy, h_new), self.by)

            if self.output_size == 1:
                y = [sigmoid(y_raw[0])]
            else:
                y = softmax(y_raw)

            cache.append((x, h, z, h_new, y))
            h = h_new

        return cache

    def bptt(self, cache, targets, mode='last'):
        '''
        Backpropagation through time.
              '''
        T  = len(cache)
        hs = self.hidden_size
        os = self.output_size
        ins = self.input_size

        dWx = [[0.0] * ins  for _ in range(hs)]
        dWh = [[0.0] * hs   for _ in range(hs)]
        dbh = [0.0] * hs
        dWy = [[0.0] * hs   for _ in range(os)]
        dby = [0.0] * os

        dh_next = [0.0] * hs
        total_loss = 0.0

        for t in reversed(range(T)):
            x, h_prev, z, h, y = cache[t]

            if mode == 'last' and t < T - 1:
                dy = [0.0] * os
            else:
                target = targets[t] if mode == 'all' else targets
                if os == 1:
                    dy = [y[0] - target]
                    total_loss += -(target * log(max(1e-12, y[0])) +
                                    (1 - target) * log(max(1e-12, 1 - y[0])))
                else:
                    dy = [y[k] - target[k] for k in range(os)]
                    total_loss += -sum(target[k] * log(max(1e-12, y[k]))
                                       for k in range(os))

            for j in range(os):
                dby[j]  += dy[j]
                for k in range(hs):
                    dWy[j][k] += dy[j] * h[k]

            dh = [0.0] * hs
            for k in range(hs):
                for j in range(os):
                    dh[k] += self.Wy[j][k] * dy[j]
                dh[k] += dh_next[k]

            dz = [dh[k] * tanh_d(z[k]) for k in range(hs)]
            dz = [clip(d, -self.clip_val, self.clip_val) for d in dz]

            for k in range(hs):
                dbh[k] += dz[k]
                for i in range(ins):
                    dWx[k][i] += dz[k] * x[i]
                for i in range(hs):
                    dWh[k][i] += dz[k] * h_prev[i]

            dh_next = [sum(self.Wh[k][i] * dz[k] for k in range(hs))
                       for i in range(hs)]

        for k in range(hs):
            dbh[k] = clip(dbh[k], -self.clip_val, self.clip_val)
            for i in range(ins):
                dWx[k][i] = clip(dWx[k][i], -self.clip_val, self.clip_val)
                self.Wx[k][i] -= self.lr * dWx[k][i]
            for i in range(hs):
                dWh[k][i] = clip(dWh[k][i], -self.clip_val, self.clip_val)
                self.Wh[k][i] -= self.lr * dWh[k][i]
            self.bh[k] -= self.lr * dbh[k]

        for j in range(os):
            self.by[j] -= self.lr * dby[j]
            for k in range(hs):
                dWy[j][k] = clip(dWy[j][k], -self.clip_val, self.clip_val)
                self.Wy[j][k] -= self.lr * dWy[j][k]

        return total_loss

    def fit(self, sequences, targets, epochs=50, mode='last'):
        self.loss_history = []
        for epoch in range(epochs):
            total = 0.0
            for seq, tgt in zip(sequences, targets):
                cache = self.forward(seq)
                loss  = self.bptt(cache, tgt, mode=mode)
                total += loss
            avg = total / len(sequences)
            self.loss_history.append(avg)
            if (epoch + 1) % 10 == 0:
                print(f'  epoch {epoch+1:>3}  loss={avg:.4f}')
        return self

    def predict(self, sequences, mode='last'):
        preds = []
        for seq in sequences:
            cache = self.forward(seq)
            if mode == 'last':
                y = cache[-1][4]
            else:
                y = [step[4] for step in cache]
            preds.append(y)
        return preds


class LSTM:
    def __init__(self, input_size, hidden_size, output_size,
                 lr=0.01, clip_val=5.0, seed=42):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr          = lr
        self.clip_val    = clip_val

        rng = LCG(seed)
        s   = sqrt(1.0 / hidden_size)
        combined = input_size + hidden_size

        def make_W():
            return [[rng.next_gaussian() * s for _ in range(combined)]
                    for _ in range(hidden_size)]
        def make_b():
            return [0.0] * hidden_size

        self.Wf, self.bf = make_W(), make_b()
        self.Wi, self.bi = make_W(), make_b()
        self.Wg, self.bg = make_W(), make_b()
        self.Wo, self.bo = make_W(), make_b()

        self.Wy = [[rng.next_gaussian() * s for _ in range(hidden_size)]
                   for _ in range(output_size)]
        self.by = [0.0] * output_size

    def _mat_vec(self, M, v):
        return [sum(M[i][j] * v[j] for j in range(len(v)))
                for i in range(len(M))]

    def _vec_add(self, a, b):
        return [a[i] + b[i] for i in range(len(a))]

    def forward(self, sequence):
        h = [0.0] * self.hidden_size
        c = [0.0] * self.hidden_size
        cache = []

        for x in sequence:
            xh = x + h  

            zf = self._vec_add(self._mat_vec(self.Wf, xh), self.bf)
            zi = self._vec_add(self._mat_vec(self.Wi, xh), self.bi)
            zg = self._vec_add(self._mat_vec(self.Wg, xh), self.bg)
            zo = self._vec_add(self._mat_vec(self.Wo, xh), self.bo)

            f = [sigmoid(z) for z in zf]
            i = [sigmoid(z) for z in zi]
            g = [tanh(z)    for z in zg]
            o = [sigmoid(z) for z in zo]

            c_new = [f[k]*c[k] + i[k]*g[k] for k in range(self.hidden_size)]
            h_new = [o[k]*tanh(c_new[k])    for k in range(self.hidden_size)]

            y_raw = self._vec_add(self._mat_vec(self.Wy, h_new), self.by)
            y = [sigmoid(y_raw[0])] if self.output_size == 1 else softmax(y_raw)

            cache.append((x, h, c, zf, zi, zg, zo, f, i, g, o, c_new, h_new, y))
            h, c = h_new, c_new

        return cache

    def bptt(self, cache, targets, mode='last'):
        T   = len(cache)
        hs  = self.hidden_size
        ins = self.input_size
        os  = self.output_size

        dWf = [[0.0]*(ins+hs) for _ in range(hs)]
        dWi = [[0.0]*(ins+hs) for _ in range(hs)]
        dWg = [[0.0]*(ins+hs) for _ in range(hs)]
        dWo = [[0.0]*(ins+hs) for _ in range(hs)]
        dbf = [0.0]*hs; dbi = [0.0]*hs
        dbg = [0.0]*hs; dbo = [0.0]*hs
        dWy = [[0.0]*hs for _ in range(os)]
        dby = [0.0]*os

        dh_next = [0.0]*hs
        dc_next = [0.0]*hs
        total_loss = 0.0

        for t in reversed(range(T)):
            x, h_prev, c_prev, zf, zi, zg, zo, f, i, g, o, c, h, y = cache[t]

            if mode == 'last' and t < T - 1:
                dy = [0.0]*os
            else:
                target = targets[t] if mode == 'all' else targets
                if os == 1:
                    dy = [y[0] - target]
                    total_loss += -(target * log(max(1e-12, y[0])) +
                                    (1-target) * log(max(1e-12, 1-y[0])))
                else:
                    dy = [y[k] - target[k] for k in range(os)]
                    total_loss += -sum(target[k]*log(max(1e-12, y[k]))
                                       for k in range(os))

            for j in range(os):
                dby[j] += dy[j]
                for k in range(hs):
                    dWy[j][k] += dy[j] * h[k]

            dh = [sum(self.Wy[j][k]*dy[j] for j in range(os)) + dh_next[k]
                  for k in range(hs)]

            do  = [dh[k] * tanh(c[k])         for k in range(hs)]
            dc  = [dh[k] * o[k] * tanh_d(c[k]) + dc_next[k] for k in range(hs)]
            df  = [dc[k] * c_prev[k]           for k in range(hs)]
            di  = [dc[k] * g[k]                for k in range(hs)]
            dg  = [dc[k] * i[k]                for k in range(hs)]
            dc_prev = [dc[k] * f[k]            for k in range(hs)]

            dzf = [clip(df[k]*(f[k]*(1-f[k])), -self.clip_val, self.clip_val) for k in range(hs)]
            dzi = [clip(di[k]*(i[k]*(1-i[k])), -self.clip_val, self.clip_val) for k in range(hs)]
            dzg = [clip(dg[k]*(1-g[k]*g[k]),   -self.clip_val, self.clip_val) for k in range(hs)]
            dzo = [clip(do[k]*(o[k]*(1-o[k])), -self.clip_val, self.clip_val) for k in range(hs)]

            xh = x + h_prev
            for k in range(hs):
                dbf[k] += dzf[k]; dbi[k] += dzi[k]
                dbg[k] += dzg[k]; dbo[k] += dzo[k]
                for j in range(ins+hs):
                    dWf[k][j] += dzf[k]*xh[j]
                    dWi[k][j] += dzi[k]*xh[j]
                    dWg[k][j] += dzg[k]*xh[j]
                    dWo[k][j] += dzo[k]*xh[j]

            dh_next = [sum((self.Wf[k][ins+m]*dzf[k] +
                            self.Wi[k][ins+m]*dzi[k] +
                            self.Wg[k][ins+m]*dzg[k] +
                            self.Wo[k][ins+m]*dzo[k])
                           for k in range(hs))
                       for m in range(hs)]
            dc_next = dc_prev

        def update(W, dW, b, db):
            for k in range(len(W)):
                b[k] -= self.lr * clip(db[k], -self.clip_val, self.clip_val)
                for j in range(len(W[k])):
                    W[k][j] -= self.lr * clip(dW[k][j], -self.clip_val, self.clip_val)

        update(self.Wf, dWf, self.bf, dbf)
        update(self.Wi, dWi, self.bi, dbi)
        update(self.Wg, dWg, self.bg, dbg)
        update(self.Wo, dWo, self.bo, dbo)

        for j in range(os):
            self.by[j] -= self.lr * dby[j]
            for k in range(hs):
                self.Wy[j][k] -= self.lr * clip(dWy[j][k], -self.clip_val, self.clip_val)

        return total_loss

    def fit(self, sequences, targets, epochs=50, mode='last'):
        self.loss_history = []
        for epoch in range(epochs):
            total = 0.0
            for seq, tgt in zip(sequences, targets):
                cache = self.forward(seq)
                total += self.bptt(cache, tgt, mode=mode)
            avg = total / len(sequences)
            self.loss_history.append(avg)
            if (epoch+1) % 10 == 0:
                print(f'  epoch {epoch+1:>3}  loss={avg:.4f}')
        return self

    def predict(self, sequences, mode='last'):
        preds = []
        for seq in sequences:
            cache = self.forward(seq)
            y = cache[-1][13] if mode == 'last' else [s[13] for s in cache]
            preds.append(y)
        return preds



def gen_sine_sequence(n=200, seq_len=20, seed=1):
    rng = LCG(seed)
    X, Y = [], []
    for _ in range(n):
        offset = rng.next_float(0, 2*PI)
        freq   = rng.next_float(0.5, 2.0)
        seq    = []
        for t in range(seq_len + 1):
            seq.append(sin(offset + freq * t * 0.3))
        x = [[seq[t]] for t in range(seq_len)]
        y = 1.0 if seq[seq_len] > seq[seq_len-1] else 0.0
        X.append(x); Y.append(y)
    return X, Y

def gen_sum_sequence(n=200, seq_len=10, seed=2):
    rng = LCG(seed)
    X, Y = [], []
    for _ in range(n):
        seq = [[1.0 if rng.next_float() > 0.5 else 0.0]
               for _ in range(seq_len)]
        total = sum(s[0] for s in seq)
        X.append(seq)
        Y.append(1.0 if total > seq_len / 2 else 0.0)
    return X, Y

def gen_copy_task(n=200, seq_len=5, seed=3):
    rng  = LCG(seed)
    X, Y = [], []
    for _ in range(n):
        signal = [1.0 if rng.next_float() > 0.5 else 0.0
                  for _ in range(seq_len)]
        seq    = [[v] for v in signal] + [[0.0]] * seq_len
        target = [0.0] * seq_len + signal
        X.append(seq)
        Y.append(target)
    return X, Y

def gen_trend_sequence(n=300, seq_len=15, seed=4):
    rng = LCG(seed)
    X, Y = [], []
    for _ in range(n):
        trend = 1 if rng.next_float() > 0.5 else -1
        val   = rng.next_float(0.3, 0.7)
        seq   = []
        for _ in range(seq_len):
            val += trend * 0.05 + rng.next_gaussian() * 0.05
            val  = max(0.0, min(1.0, val))
            seq.append([val])
        X.append(seq)
        Y.append(1.0 if trend == 1 else 0.0)
    return X, Y



sep = '='*52

def eval_binary(model, X, Y):
    preds = model.predict(X, mode='last')
    correct = 0
    for p, y in zip(preds, Y):
        val = p[0] if isinstance(p[0], float) else p[0][0]
        pred_label = 1 if val >= 0.5 else 0
        true_label = int(y)
        if pred_label == true_label:
            correct += 1
    return correct / len(Y)

print(sep)
print('TEST 1 — RNN: sum sequence (accumulate over time)')
print(sep)
X, Y = gen_sum_sequence(n=300, seq_len=10)
split = int(len(X)*0.8)
Xtr,Xte,Ytr,Yte = X[:split],X[split:],Y[:split],Y[split:]
print()
rnn1 = RNN(input_size=1, hidden_size=8, output_size=1, lr=0.05)
rnn1.fit(Xtr, Ytr, epochs=50)
acc = eval_binary(rnn1, Xte, Yte)
print(f'  Test accuracy: {acc*100:.2f}%')

print()
print(sep)
print('TEST 2 — RNN: trend detection')
print(sep)
X2, Y2 = gen_trend_sequence(n=300, seq_len=15)
split2  = int(len(X2)*0.8)
Xtr2,Xte2,Ytr2,Yte2 = X2[:split2],X2[split2:],Y2[:split2],Y2[split2:]
print()
rnn2 = RNN(input_size=1, hidden_size=16, output_size=1, lr=0.02)
rnn2.fit(Xtr2, Ytr2, epochs=50)
acc2 = eval_binary(rnn2, Xte2, Yte2)
print(f'  Test accuracy: {acc2*100:.2f}%')

print()
print(sep)
print('TEST 3 — RNN: sine wave direction prediction')
print(sep)
X3, Y3 = gen_sine_sequence(n=300, seq_len=10)
split3  = int(len(X3)*0.8)
Xtr3,Xte3,Ytr3,Yte3 = X3[:split3],X3[split3:],Y3[:split3],Y3[split3:]
print()
rnn3 = RNN(input_size=1, hidden_size=16, output_size=1, lr=0.02)
rnn3.fit(Xtr3, Ytr3, epochs=50)
acc3 = eval_binary(rnn3, Xte3, Yte3)
print(f'  Test accuracy: {acc3*100:.2f}%')

print()
print(sep)
print('TEST 4 — LSTM: sum sequence (compare vs RNN)')
print(sep)
print()
lstm1 = LSTM(input_size=1, hidden_size=8, output_size=1, lr=0.05)
lstm1.fit(Xtr, Ytr, epochs=50)
acc_lstm = eval_binary(lstm1, Xte, Yte)
print(f'  RNN  accuracy : {acc*100:.2f}%')
print(f'  LSTM accuracy : {acc_lstm*100:.2f}%')

print()
print(sep)
print('TEST 5 — LSTM: trend detection')
print(sep)
print()
lstm2 = LSTM(input_size=1, hidden_size=16, output_size=1, lr=0.02)
lstm2.fit(Xtr2, Ytr2, epochs=50)
acc_lstm2 = eval_binary(lstm2, Xte2, Yte2)
print(f'  RNN  accuracy : {acc2*100:.2f}%')
print(f'  LSTM accuracy : {acc_lstm2*100:.2f}%')

print()
print(sep)
print('TEST 6 — RNN many-to-many: copy task')
print('  Show sequence, predict it back step by step')
print(sep)
X6, Y6 = gen_copy_task(n=200, seq_len=5)
split6  = int(len(X6)*0.8)
Xtr6,Xte6,Ytr6,Yte6 = X6[:split6],X6[split6:],Y6[:split6],Y6[split6:]
print()
rnn6 = RNN(input_size=1, hidden_size=16, output_size=1, lr=0.05)
rnn6.fit(Xtr6, Ytr6, epochs=80, mode='all')
preds6 = rnn6.predict(Xte6[:5], mode='all')
print('  Sample predictions vs true (last 5 steps = copy):')
for k in range(5):
    true_seq  = [round(v, 0) for v in Yte6[k][5:]]
    pred_seq  = [round(p[0], 0) for p in preds6[k][5:]]
    print(f'  true={true_seq}  pred={pred_seq}')
