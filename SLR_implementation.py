'''
y = mx + c (simple linear equation)
or
y = β0 + β1Xi
Sum_of_Squared_Errors(S.S.E)= (Yi - Y_hat)²
β0 = Y_bar - β1 * X_bar
β1 = Σ(Yi - Y_bar)(Xi - X_bar) / Σ(Xi - X_bar)²
'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

X = [1, 2, 3, 4, 5, 6]
Y = [2.1, 3.8, 6.2, 7.9, 10.3, 11.8]


def SimpleLinearRegression(X, Y):

    X_bar = sum(X) / len(X)
    Y_bar = sum(Y) / len(Y)

    temp_a = []
    temp_b = []

    i = 0
    while i < len(X):
        a = (Y[i] - Y_bar) * (X[i] - X_bar)
        b = (X[i] - X_bar) ** 2
        temp_a.append(a)
        temp_b.append(b)
        i += 1

    a = sum(temp_a)
    b = sum(temp_b)

    β1 = a / b
    β0 = Y_bar - β1 * X_bar

    y_hat = []
    for x in X:
        y_hat_i = β0 + β1 * x
        y_hat.append(y_hat_i)

    SSE_list = []
    for i in range(len(Y)):
        SSE_list.append((Y[i] - y_hat[i]) ** 2)

    t_SSE_list = []
    for i in range(len(Y)):
        t_SSE_list.append((Y[i] - Y_bar) ** 2)

    n = len(SSE_list)
    SSE = sum(SSE_list)
    t_SSE = sum(t_SSE_list)
    MSE = SSE / n
    r2_score = 1 - SSE / t_SSE

    print(f"β1       : {β1:.4f}")
    print(f"β0       : {β0:.4f}")
    print(f"y_hat    : {[round(v,3) for v in y_hat]}")
    print(f"SSE      : {SSE:.4f}")
    print(f"MSE      : {MSE:.4f}")
    print(f"R²       : {r2_score:.4f}")

    return β0, β1, y_hat, SSE_list, Y_bar, r2_score, MSE


β0, β1, y_hat, SSE_list, Y_bar, r2_score, MSE = SimpleLinearRegression(X, Y)


x_line = [min(X) - 0.3, max(X) + 0.3]
y_line = [β0 + β1 * x for x in x_line]

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Simple Linear Regression — From Scratch", fontsize=15, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)


ax1 = fig.add_subplot(gs[0, :])   # full top row
ax1.scatter(X, Y, color='royalblue', zorder=5, s=80, label='Actual Y')
ax1.plot(x_line, y_line, color='tomato', linewidth=2,
         label=f'Regression line: ŷ = {β0:.3f} + {β1:.3f}·X')

# draw residual lines (errors)
for i in range(len(X)):
    ax1.plot([X[i], X[i]], [Y[i], y_hat[i]],
             color='gray', linestyle='--', linewidth=1)

for i in range(len(X)):
    ax1.annotate(f'  ({X[i]}, {Y[i]})', (X[i], Y[i]), fontsize=8, color='navy')

ax1.axhline(Y_bar, color='green', linestyle=':', linewidth=1.2,
            label=f'Ȳ = {Y_bar:.2f}')
ax1.set_title("Regression Line with Residuals (dashed = error)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()
ax1.grid(True, alpha=0.3)


ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(X, Y,     'o-', color='royalblue', label='Actual Y',    linewidth=1.5)
ax2.plot(X, y_hat, 's--', color='tomato',   label='Predicted ŷ', linewidth=1.5)
ax2.set_title("Actual vs Predicted")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.legend()
ax2.grid(True, alpha=0.3)


ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(X, SSE_list, color='salmon', edgecolor='darkred', width=0.5)
for i, v in enumerate(SSE_list):
    ax3.text(X[i], v + 0.002, f'{v:.4f}', ha='center', fontsize=8)
ax3.set_title(f"Squared Error per Point  |  SSE={sum(SSE_list):.4f}  R²={r2_score:.4f}")
ax3.set_xlabel("X")
ax3.set_ylabel("(Yi − ŷi)²")
ax3.grid(True, alpha=0.3, axis='y')


plt.savefig("SLR_visualization.png", dpi=150, bbox_inches='tight')
print("\nPlot saved as SLR_visualization.png")
plt.show()