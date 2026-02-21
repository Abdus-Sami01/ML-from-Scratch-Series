'''
In Multiple linear regression, there are multiple features and a single target.
the equation looks like Y = β0 + β1X1 + β2X2 + ....... + βnXn

or can be written simply as Y = Xθ
where theta is the model's parameter accounting for the regression line. 

this can simplified using some mathematical steps to: θ = (XᵀX)⁻¹(XᵀY) 
Idea is simply just differentiating loss function with respect to θ and setting it to zero for global minimum.
'''

X = [[1,2,3,4,5,6],
     [1.1,2.2,3.3,4.4,5.5,6.6]]
Y = [1.1, 4.4, 9.9, 15.6, 27.5, 39.6]


class MultipleLinearRegression:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.theta = []

    def add_bias(self, X):
        X_bias = []
        for i in range(len(X[0])):        
            row = [1]                     
            for j in range(len(X)):      
                row.append(X[j][i])
            X_bias.append(row)
        return X_bias                      

    def transpose(self, X):
        rows = len(X)
        cols = len(X[0])
        XT = []
        for j in range(cols):
            row = []
            for i in range(rows):
                row.append(X[i][j])
            XT.append(row)
        return XT                          

    def mat_multiply(self, A, B):
        rows_A = len(A)
        cols_A = len(A[0])
        cols_B = len(B[0])
        result = []
        for i in range(rows_A):
            row = []
            for j in range(cols_B):
                dot = 0
                for k in range(cols_A):
                    dot += A[i][k] * B[k][j]   
                row.append(dot)
            result.append(row)
        return result

    def mat_inverse(self, M):
        n = len(M)
        augmented = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(float(M[i][j]))
            for j in range(n):
                row.append(1.0 if i == j else 0.0)
            augmented.append(row)

        for col in range(n):
            pivot = augmented[col][col]
            for j in range(2 * n):
                augmented[col][j] = augmented[col][j] / pivot
            for row in range(n):
                if row != col:
                    factor = augmented[row][col]
                    for j in range(2 * n):
                        augmented[row][j] -= factor * augmented[col][j]

        inverse = []
        for i in range(n):
            row = []
            for j in range(n, 2 * n):
                row.append(augmented[i][j])
            inverse.append(row)
        return inverse

    def mat_vec_multiply(self, M, v):
        result = []
        for i in range(len(M)):
            dot = 0
            for j in range(len(v)):
                dot += M[i][j] * v[j]
            result.append(dot)
        return result

    def fit(self):
        X_bias  = self.add_bias(self.X)          # (n × p+1)
        XT      = self.transpose(X_bias)          # (p+1 × n)
        XTX     = self.mat_multiply(XT, X_bias)   # (p+1 × p+1)
        XTX_inv = self.mat_inverse(XTX)           # (p+1 × p+1)
        XTY     = self.mat_vec_multiply(XT, self.Y) # (p+1,)
        self.theta = self.mat_vec_multiply(XTX_inv, XTY) # (p+1,)

        print("\n── Model Parameters ──")
        print(f"β0 (bias) : {self.theta[0]:.4f}")
        for i in range(1, len(self.theta)):
            print(f"β{i}        : {self.theta[i]:.4f}")

    def predict(self, X):
        X_bias = self.add_bias(X)
        y_hat  = []
        for i in range(len(X_bias)):
            dot = 0
            for j in range(len(self.theta)):
                dot += self.theta[j] * X_bias[i][j]
            y_hat.append(dot)
        return y_hat

    def evaluate(self, y_hat):
        Y     = self.Y
        Y_bar = sum(Y) / len(Y)
        n     = len(Y)

        SSE_list = []
        for i in range(n):
            SSE_list.append((Y[i] - y_hat[i]) ** 2)

        SST_list = []
        for i in range(n):
            SST_list.append((Y[i] - Y_bar) ** 2)

        SSE = sum(SSE_list)
        SST = sum(SST_list)
        MSE = SSE / n
        R2  = 1 - SSE / SST

        print("\n── Evaluation ──")
        print(f"y_hat : {[round(v, 3) for v in y_hat]}")
        print(f"SSE   : {SSE:.4f}")
        print(f"MSE   : {MSE:.4f}")
        print(f"R²    : {R2:.4f}")


mlr = MultipleLinearRegression(X, Y)
mlr.fit()
y_hat = mlr.predict(X)
mlr.evaluate(y_hat)