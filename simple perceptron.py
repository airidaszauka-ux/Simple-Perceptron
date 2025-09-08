import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Perceptron with Teach + Fine-tune phases
# ============================================

# --- Hardcoded dataset (x1, x2, d) ---
data_teach = np.array([
    [0.21835, 0.81884,  1],
    [0.14115, 0.83535,  1],
    [0.37022, 0.81110,  1],
    [0.36484, 0.85180,  1],
    [0.55223, 0.83449,  1],
    [0.49187, 0.80889,  1],
    [0.08838, 0.62068, -1],
    [0.09817, 0.79092, -1]
])

data_tune = np.array([
    [0.31565, 0.83101,  1],
    [0.46111, 0.82518,  1],
    [0.16975, 0.84049,  1],
    [0.14913, 0.77104, -1],
    [0.18474, 0.62790, -1]
])

# Split features and labels
X_teach, d_teach = data_teach[:, :2], data_teach[:, 2]
X_tune, d_tune   = data_tune[:, :2], data_tune[:, 2]

# --- Hyperparameters ---
eta = 0.1
epochs = 200

# --- Random initialization ---
np.random.seed(42)
w = np.random.randn(2)
b = np.random.randn()

# --- Training function ---
def train_phase(X, d, w, b, eta, epochs, phase="teach"):
    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            x1, x2 = X[i]
            y = 1 if (x1*w[0] + x2*w[1] + b) > 0 else -1
            e = d[i] - y
            if e != 0:
                if phase == "teach":
                    # normal update
                    w[0] += eta * e * x1
                    w[1] += eta * e * x2
                    b    += eta * e
                elif phase == "tune":
                    # fine-tuning update (smaller adjustment)
                    w[0] += (eta/2) * e * x1
                    w[1] += (eta/2) * e * x2
                    b    += (eta/2) * e
                errors += 1
        print(f"[{phase}] Epoch {epoch+1}: errors = {errors}")
        if errors == 0:
            print(f"[{phase}] converged.")
            break
    return w, b

# --- Phase 1: Teaching ---
w, b = train_phase(X_teach, d_teach, w, b, eta, epochs, phase="teach")

# --- Phase 2: Fine-tuning ---
w, b = train_phase(X_tune, d_tune, w, b, eta, epochs, phase="tune")

# --- Accuracy report ---
X_all = np.vstack([X_teach, X_tune])
d_all = np.hstack([d_teach, d_tune])
preds = np.array([1 if (x[0]*w[0] + x[1]*w[1] + b) > 0 else -1 for x in X_all])
acc = (preds == d_all).mean()*100
print("\nFinal weights:", w)
print("Final bias:", b)
print(f"Accuracy on all data: {acc:.2f}%")

# --- Plot ---
class1 = X_all[d_all == 1]
class2 = X_all[d_all == -1]
plt.scatter(class1[:,0], class1[:,1], color='blue', marker='o', label='Class +1')
plt.scatter(class2[:,0], class2[:,1], color='red',  marker='x', label='Class -1')

if abs(w[1]) > 1e-12:
    x_vals = np.linspace(X_all[:,0].min()-0.1, X_all[:,0].max()+0.1, 100)
    y_vals = -(w[0]*x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'k--', label='Decision boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron with Teach + Fine-tune')
plt.legend()
plt.show()
