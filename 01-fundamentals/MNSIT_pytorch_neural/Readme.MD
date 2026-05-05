# PyTorch Neural Network (Day 2)

## Overview

Rebuilt the Day 1 neural network using PyTorch - same problem, same results, but with automatic gradients and cleaner code.

**Achievement: 90% accuracy with 50% less code than NumPy implementation**

---

## What I Learned

### Core Concepts

#### 1. PyTorch Tensors
```python
# Like NumPy arrays but GPU-ready
X = torch.tensor([[1.], [2.], [3.]])
y = torch.tensor([[0.], [0.], [1.]])

# Can convert between NumPy and PyTorch
numpy_array = X.numpy()
torch_tensor = torch.from_numpy(numpy_array)
```

#### 2. Automatic Gradients (Autograd)
**Day 1 (NumPy) - Manual:**
```python
# I had to compute gradients manually
error = y_pred - y_true
dW = (X.T @ error) / m
db = np.mean(error)
```

**Day 2 (PyTorch) - Automatic:**
```python
# PyTorch computes ALL gradients automatically
loss.backward()  # That's it!
```

**Key insight:** `loss.backward()` does the same math I did in Day 1, but automatically for ALL parameters.

---

#### 3. nn.Module (Model Structure)

**How weights are initialized:**
```python
# When you create:
layer = nn.Linear(input_size, output_size)

# PyTorch creates:
# - Weight matrix: (output_size, input_size)
# - Bias vector: (output_size,)
# - Random initialization using Kaiming method

# Example:
layer = nn.Linear(1, 1)
# Weight shape: (1, 1)
# Bias shape: (1,)

layer = nn.Linear(784, 128)
# Weight shape: (128, 784)
# Bias shape: (128,)
```

**My Day 1 problem:**
```python
# Day 1 (NumPy):
W = np.random.randn(1, 1) * 0.1
b = np.zeros((1, 1))

# Day 2 (PyTorch):
self.linear = nn.Linear(1, 1)  # Creates W and b automatically
```

---

#### 4. Loss Functions

**Important: Different losses have different gradient formulas!**

```python
# BCE Loss for binary classification
criterion = nn.BCELoss()
# Gradient formula: specific to BCE

# MSE Loss for regression
criterion = nn.MSELoss()
# Gradient formula: 2 * (y_pred - y_true)

# CrossEntropy Loss for multi-class
criterion = nn.CrossEntropyLoss()
# Gradient formula: specific to CE
```

**PyTorch automatically uses the correct gradient formula based on which loss you choose.**

This is what I computed manually in Day 1:
```python
# Day 1: I had to know BCE gradient formula
error = y_pred - y_true  # This is specific to BCE!

# Day 2: PyTorch knows all formulas
loss.backward()  # Computes correct gradient automatically
```

---

#### 5. Optimizers (SGD)

**What is SGD?**
```python
# SGD = Stochastic Gradient Descent
# Update rule: θ_new = θ_old - learning_rate * gradient

# Day 1 (NumPy) - I did this manually:
W = W - lr * dW
b = b - lr * db

# Day 2 (PyTorch) - Same thing, automated:
optimizer = optim.SGD(model.parameters(), lr=0.5)
optimizer.step()  # Does: param = param - lr * param.grad
```

**I WAS already doing SGD in Day 1 - just didn't call it that!**

---

#### 6. Training Loop Pattern

```python
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()  # Clear old gradients (important!)
    loss.backward()         # Compute gradients automatically
    optimizer.step()        # Update weights: W = W - lr*dW
```

**Why `optimizer.zero_grad()`?**
- Gradients ACCUMULATE by default
- Without `.zero_grad()`, new gradients ADD to old ones
- We want fresh gradients each iteration

---

## Code Comparison

### Day 1 (NumPy) - Manual Everything
```python
# ~100 lines of code

def forward(X, W, b):
    z = X @ W + b
    return sigmoid(z)

def compute_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + 
                    (1 - y_true) * np.log(1 - y_pred))
    return loss

def compute_gradients(X, y_true, y_pred):
    error = y_pred - y_true
    dW = (X.T @ error) / m
    db = np.mean(error)
    return dW, db

def update_weights(W, b, dW, db, lr):
    W = W - lr * dW
    b = b - lr * db
    return W, b

# Training loop
for epoch in range(epochs):
    y_pred = forward(X, W, b)
    loss = compute_loss(y, y_pred)
    dW, db = compute_gradients(X, y, y_pred)
    W, b = update_weights(W, b, dW, db, lr)
```

### Day 2 (PyTorch) - Automated
```python
# ~50 lines of code

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Training loop
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**50% code reduction, same results!**

---

## Key Insights

### What PyTorch Automates:
✅ Gradient computation (`loss.backward()`)
✅ Parameter updates (`optimizer.step()`)
✅ Loss functions (`nn.BCELoss()`)
✅ Weight initialization (`nn.Linear()`)

### What Hasn't Changed:
- The underlying algorithm (still gradient descent)
- The mathematics (still calculus and linear algebra)
- The training process (forward → loss → backward → update)

### Why Day 1 Was Important:
- I understand what `loss.backward()` actually does
- I can debug when things go wrong
- I'm not just using "magic tools"
- I know the fundamentals

### Why Day 2 Is Powerful:
- Can build complex models quickly
- Easy to experiment with different architectures
- Industry standard framework
- GPU acceleration ready

---

## Results

**Both implementations achieve:**
- Final loss: ~0.02
- Decision boundary: ~5.4
- Accuracy: 90%

**Proof that PyTorch does the same math, just automated!**

---

## Files

- `pytorch_neural_network.py` - Complete implementation
- `pytorch_final_results.png` - Training visualization
- `loss_optimizer_guide.md` - Quick reference for losses and optimizers

---

## Common Issues I Faced

### Issue 1: Understanding Weight Shapes
**Question:** Does `nn.Linear(1, 1)` create a scalar or matrix?

**Answer:** It creates:
- Weight: (1, 1) matrix
- Bias: (1,) vector

For `nn.Linear(input_size, output_size)`:
- Weight: (output_size, input_size)
- Bias: (output_size,)

---

### Issue 2: Loss Function Choice
**Question:** Does loss type affect gradient formula?

**Answer:** YES! Different losses produce different gradients:
- BCE: gradient specific to BCE
- MSE: gradient = 2 * (y_pred - y_true)
- PyTorch uses correct formula automatically

---

### Issue 3: What is SGD?
**Question:** We didn't use SGD in NumPy, what does it do?

**Answer:** I WAS using SGD in NumPy!
```python
W = W - lr * dW  # This IS SGD
```
PyTorch's `optimizer.step()` does the same thing.

---

## Next Steps

✅ Completed: Single-layer network in PyTorch
→ Next: MNIST (real dataset, multi-class classification)
→ Then: CNNs (convolutional layers for images)

---

**Time spent:** 3 hours

**Date:** March 2026

**Status:** ✅ PyTorch fundamentals mastered

**Key takeaway:** PyTorch automates what I built in Day 1, but the fundamentals remain the same.
