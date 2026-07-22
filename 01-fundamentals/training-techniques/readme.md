In this we will explore all training techniques and there mathematical aspects
# Day 5: Training Techniques

Deep dive into four techniques that stabilize training and fight overfitting:
**Batch Normalization, Dropout, Weight Decay, and Learning Rate Scheduling + Early Stopping.**

This README covers the syntax, the math (with worked numeric examples), and *why* each
technique is placed where it is in the architecture — not just how to use it.

---

## Setup Used for All Experiments

To make overfitting visible, we deliberately trained on a **small subset (2000 samples)**
of MNIST instead of the full 60,000. A small dataset relative to model capacity is what
causes a model to memorize instead of generalize — that gap (train accuracy vs test
accuracy) is what every technique below is trying to shrink.

```python
small_train = Subset(train_dataset, range(2000))
train_loader = DataLoader(small_train, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

**Key lesson learned:** full data + few epochs does NOT reveal overfitting — the model
barely sees any image twice, so it can't memorize. Overfitting needs *repeated exposure*
to a *limited* dataset. Small subset + more epochs is the correct setup to observe it.

---

## 1. Batch Normalization

### The Problem It Solves

As data flows through layers, activation values can drift to very different scales at
each layer (this shift is called **Internal Covariate Shift**). This causes:

- Unstable/jumpy gradients during backprop
- Vanishing gradients when activations get very small
- Each layer having to constantly readjust to the shifting scale coming from the
  previous layer, which itself keeps changing as earlier weights update during training

BatchNorm's job: keep activations in a consistent, predictable range (mean ≈ 0, std ≈ 1)
at every layer, so training stays smooth and stable.

### What Gets Normalized — Concretely

For a batch of 32 images passing through `conv1` (32 filters):

```
Output shape = (batch_size, channels, height, width) = (32, 32, 28, 28)
```

`BatchNorm2d` computes statistics **per channel**, pooling across *all images in the
batch* AND *all spatial positions* in that channel.

**Why pool across spatial positions?** Every position in one channel's 28×28 feature map
was produced by the *same filter* — e.g. if channel 5 is a vertical-edge detector, every
one of its 784 positions is answering the same question ("edge here?"), just at
different locations. They're the same kind of measurement, so pooling their statistics
together is valid.

Contrast: pooling channel 5 (edge detector) together with channel 12 (maybe a corner
detector) would mix two *different* measurements — like averaging temperature with
pressure. That's why normalization is done **per channel**, never across channels.

For one channel, that's `32 images × 28 × 28 = 25,088` values reduced to just **one mean
and one variance** for that channel.

### The Formula

```
Step 1 — batch statistics (per channel):
  μ_B = mean of all values in this channel across the batch
  σ²_B = variance of all values in this channel across the batch

Step 2 — normalize:
  x̂ = (x − μ_B) / sqrt(σ²_B + ε)        (ε ≈ 1e-5, prevents divide-by-zero)

Step 3 — scale and shift (LEARNABLE):
  y = γ·x̂ + β
```

### Worked Numeric Example

Channel 5's raw output across a tiny batch (2 images, 2×2 feature map each):

```
Image 1:  [5.0, 7.0]        Image 2:  [6.0, 4.0]
          [3.0, 9.0]                  [8.0, 2.0]

All 8 values: [5.0, 7.0, 3.0, 9.0, 6.0, 4.0, 8.0, 2.0]
```

**Step 1 — statistics:**
```
μ_B = (5+7+3+9+6+4+8+2)/8 = 44/8 = 5.5

σ²_B = mean of (x − 5.5)² for each x
     = [0.25+2.25+6.25+12.25+0.25+2.25+6.25+12.25] / 8
     = 42/8 = 5.25
```

**Step 2 — normalize** (`x̂ = (x−5.5)/√5.25 ≈ (x−5.5)/2.29`):
```
5.0 → -0.22    7.0 →  0.65    3.0 → -1.09    9.0 →  1.53
6.0 →  0.22    4.0 → -0.65    8.0 →  1.09    2.0 → -1.53

Result: mean ≈ 0, std ≈ 1
```

**Step 3 — scale/shift** (say learned γ₅=1.2, β₅=0.3): `y = 1.2·x̂ + 0.3`
```
x̂=-0.22 → y = 1.2(-0.22)+0.3 = 0.036
x̂= 0.65 → y = 1.2(0.65)+0.3  = 1.08
... (repeated for all 8 values)
```

This final `y` — not `x̂` — is what gets passed to ReLU next.

### γ and β Are Real, Learnable Weights

This was a key realization: **γ and β are not special magic values** — they are
learnable parameters trained by backprop + gradient descent, exactly like conv/linear
weights.

```python
bn = nn.BatchNorm2d(32)
print(bn.weight.shape)   # torch.Size([32])  <- this IS gamma
print(bn.bias.shape)     # torch.Size([32])  <- this IS beta
print(bn.weight[:5])     # tensor([1., 1., 1., 1., 1.])  (init to 1)
print(bn.bias[:5])       # tensor([0., 0., 0., 0., 0.])  (init to 0)
```

PyTorch literally stores γ as `.weight` and β as `.bias` inside the BatchNorm module —
same naming convention as every other layer, because they update through the exact same
mechanism: `loss.backward()` computes `dLoss/dγ` and `dLoss/dβ`, and `optimizer.step()`
updates them like any other weight. `model.parameters()` includes them automatically.

**Why γ, β exist at all:** pure normalization forces every layer's output to mean=0,
std=1 — a rigid constraint. γ and β give the network the *freedom* to shift/scale back
to whatever distribution is actually useful for that feature, while still getting the
stability benefit of normalization.

### train() vs eval() — Why It Matters Here

```
model.train():
  BatchNorm uses the CURRENT BATCH's statistics (μ_B, σ²_B computed fresh)

model.eval():
  BatchNorm uses a RUNNING AVERAGE of statistics accumulated during training
  (because at inference you might get just 1 image — can't compute
  meaningful "batch statistics" from a single sample)
```

Forgetting `model.eval()` before testing is a classic bug — BatchNorm silently uses the
wrong statistics and results become inconsistent.

### Where It's Placed and Why

```python
# Correct order:
x = self.pool(self.relu(self.bn1(self.conv1(x))))
#              Conv → BatchNorm → ReLU → Pool
```

- **After Conv/Linear, before activation** — we normalize the raw, unnormalized signal
  before squashing it through ReLU. Normalizing *after* ReLU means working with an
  already-distorted (non-negative-only) distribution.
- **Not on the input layer** — pixel data is already normalized by `ToTensor()` (0–1
  range).
- **Not on the output layer** — CrossEntropyLoss needs raw logits; renormalizing final
  scores would distort the loss computation.

### Syntax

```python
self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
self.bn1   = nn.BatchNorm2d(32)     # 2D for conv layers — must match conv1's out_channels

self.fc1   = nn.Linear(64*7*7, 128)
self.bn3   = nn.BatchNorm1d(128)    # 1D for FC layers — must match fc1's out_features
```

**Bug we hit and fixed:** naming two different BatchNorm layers with the same attribute
name (`self.bn1` twice) silently *overwrites* the first one — no error until `forward()`
runs and shapes mismatch. Also: `BatchNorm1d`'s size must exactly match the *output*
size of the preceding Linear layer, not some other number.

---

## 2. Dropout

### The Problem It Solves

A network can overfit by relying heavily on specific neurons or specific *combinations*
of neurons to memorize training examples. Dropout breaks this by randomly disabling
neurons during training, forcing the network to build **redundant, distributed**
representations instead of depending on any single neuron.

### The Formula

```
Step 1 — random mask (per neuron, fresh every forward pass):
  mask_i ~ Bernoulli(1 − p)     (1 = keep, 0 = drop, with drop probability p)

Step 2 — apply mask:
  x_dropped = x · mask

Step 3 — scale survivors:
  x_final = x_dropped / (1 − p)
```

### Worked Numeric Example (p = 0.5)

Activations after ReLU, before dropout:
```
[2.0, 0.0, 3.5, 1.2, 0.0, 4.1, 2.8, 0.9]
```

**Step 1 — random mask:**
```
mask = [1, 0, 1, 1, 0, 0, 1, 0]
```

**Step 2 — apply mask:**
```
after_mask = [2.0, 0.0, 3.5, 1.2, 0.0, 0.0, 2.8, 0.0]
```

**Step 3 — scale by 1/(1−0.5) = 2.0:**
```
final = [4.0, 0.0, 7.0, 2.4, 0.0, 0.0, 5.6, 0.0]
```

### Where 1/(1−p) Comes From (Expected Value Derivation)

We want the **average** output during training (with dropout) to match the **actual**
output during eval (no dropout, all neurons active).

```
For one neuron with true value x:

Without dropout (eval): output = x, so E[output] = x

With dropout (before scaling), output is a random variable:
  E[output] = (1−p)·x + p·0 = (1−p)·x
  (with probability (1−p) it survives with value x, with probability p it's 0)

We want: E[scale · output] = x
         scale · (1−p) · x = x
         scale = 1/(1−p)
```

**Numeric check (p=0.5, x=10):**
```
Without dropout: output = 10
With dropout, unscaled: E[output] = 0.5×10 + 0.5×0 = 5
Scale needed: 10/5 = 2 = 1/(1−0.5) ✓ matches formula exactly
```

**Why this matters:** without this scaling, the *sum* of activations flowing into the
next layer would be roughly half its normal magnitude during training but full
magnitude during eval (no dropout there) — a systematic mismatch between train-time and
eval-time input scale for the next layer. Scaling by `1/(1−p)` keeps expected magnitude
consistent across both modes. This exact technique is called **inverted dropout** and is
what `nn.Dropout` implements internally.

### train() vs eval()

```
model.train() → Dropout is ACTIVE (randomly zeroes + scales)
model.eval()  → Dropout does NOTHING (all neurons pass through unchanged)
```

We turn dropout off at eval time not because of weight updates (eval never updates
weights either way) — we turn it off because we want the model's **full learned
capacity** making the prediction, using every neuron together for the best, most
complete result.

### Where It's Placed and Why

```python
x = self.dropout(self.relu(self.bn3(self.fc1(x))))
#                  FC → BatchNorm → ReLU → Dropout
```

- **After activation (post-ReLU)** — dropout should zero out *activated* outputs, not
  raw pre-activation values.
- **Rarely used in conv layers** (or only lightly) — conv layers already have implicit
  regularization from weight sharing, and randomly dropping individual pixels disrupts
  the spatial correlation CNNs rely on. If used in conv layers, people use
  `nn.Dropout2d` (drops whole channels, respecting spatial structure) instead.
- **Never in the output layer** — you need all final class logits intact to make a valid
  prediction; randomly zeroing them would randomly kill predictions for certain classes.

### Syntax

```python
self.dropout = nn.Dropout(p=0.5)   # p = probability of dropping a neuron

def forward(self, x):
    x = self.dropout(self.relu(self.bn3(self.fc1(x))))
```

---

## 3. Weight Decay (L2 Regularization)

### The Problem It Solves

Large weights make a model react very sharply to small input changes:

```
w = 0.5 (small):  x=1.0 → out=0.5,  x=1.1 → out=0.55   (smooth, stable)
w = 50  (large):  x=1.0 → out=50,   x=1.1 → out=55     (huge jump from tiny input change)
```

That kind of sensitivity is exactly what memorization looks like — the model fires
precisely on very specific input values instead of learning smooth, general patterns.
Weight decay discourages weights from growing unnecessarily large.

### The Formula

```
Normal loss:        L = CrossEntropyLoss(y_pred, y_true)
With weight decay:   L_total = L + λ · Σ(w²)

  λ = weight_decay hyperparameter (e.g. 1e-4)
  Σ(w²) = sum of squares of every weight in the model
```

### Worked Numeric Example

4 weights: `[0.5, -0.3, 2.0, 0.1]`, λ = 0.01

```
Σ(w²) = 0.5² + (-0.3)² + 2.0² + 0.1²
      = 0.25 + 0.09 + 4.0 + 0.01
      = 4.35

penalty = 0.01 × 4.35 = 0.0435
L_total = L + 0.0435
```

**Why squaring punishes large weights disproportionately:** weight `2.0` alone
contributes `4.0` out of the `4.35` total — almost all of the penalty. Squaring grows
*faster than linear*: doubling a weight (e.g. 0.5→1.0) quadruples its penalty
contribution (0.25→1.0), not just doubles it. Large weights get hit much harder than
small ones.

### How It Changes the Gradient Update

```
Normal update:          w_new = w_old − lr·(dL/dw)

Derivative of penalty:  d(λw²)/dw = 2λw

Full update:             w_new = w_old − lr·(dL/dw + 2λw)
                                = w_old·(1 − 2·lr·λ) − lr·(dL/dw)
                                              ↑
                                  factor slightly less than 1 —
                                  shrinks w a little every step
```

**Numeric check** (w_old=2.0, lr=0.001, λ=1e-4, pretend dL/dw=0 to isolate the decay
effect):
```
w_new = 2.0 × (1 − 2×0.001×0.0001)
      = 2.0 × 0.9999998
      = 1.9999996
```

Tiny per-step shrinkage — but repeated over thousands of steps, it accumulates
meaningfully, **unless** the actual task gradient (`dL/dw`) keeps pushing the weight back
up because it's genuinely useful for reducing prediction error.

### The Tug-of-War

```
Force 1 (task gradient, dL/dw):  pushes weight to whatever value reduces error
Force 2 (weight decay, 2λw):     ALWAYS pulls weight toward zero, regardless of task

Final weight = wherever these two forces balance.
Genuinely useful weights survive (Force 1 wins).
Marginally useful / memorization-only weights shrink toward zero (Force 2 wins).
```

### Where It's Applied

Not a layer — it's a property of the **optimizer**, applied uniformly to every
trainable weight in the model automatically.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

---

## 4. Learning Rate Scheduling + Early Stopping

### Why a Fixed Learning Rate Is a Problem

Gradient descent is like walking downhill toward a minimum loss point.

```
Early training: far from minimum → large steps (high LR) cover distance fast
Late training:  close to minimum → large steps OVERSHOOT, bouncing back and forth,
                                    never settling precisely
```

**Concrete example** — minimum at w=5.0, current w=4.8, LR=0.5:
```
Step: w = 4.8 + 0.5 = 5.3   → overshot past 5.0!
Next step (gradient reverses): w = 5.3 − 0.5 = 4.8 → back where we started
→ oscillates forever, never converges precisely

With smaller LR=0.05: 4.8 → 4.85 → 4.90 → 4.95 → 4.99 → smoothly converges to 5.0
```

### StepLR (Fixed Schedule)

```
Formula:  new_lr = initial_lr × gamma^floor(epoch / step_size)
```

Example: `initial_lr=0.1, step_size=5, gamma=0.5`
```
Epoch 1–4:   lr = 0.1 × 0.5⁰ = 0.1
Epoch 5–9:   lr = 0.1 × 0.5¹ = 0.05
Epoch 10–14: lr = 0.1 × 0.5² = 0.025
Epoch 15–19: lr = 0.1 × 0.5³ = 0.0125
```

Halves every 5 epochs, blindly, on a timer — regardless of how training is actually
going.

### ReduceLROnPlateau (Adaptive — What We Used)

Watches actual performance instead of a fixed timer.

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)
```

```
mode='max'   → tracking something to MAXIMIZE (test accuracy)
factor=0.5   → when triggered, multiply LR by 0.5
patience=3   → wait 3 epochs of no improvement before reducing
```

**Mechanism, step by step:**
```
Epoch 1: acc=90.0%  → best=90.0%, counter=0
Epoch 2: acc=91.5%  → improved! best=91.5%, counter=0
Epoch 3: acc=91.3%  → no improvement → counter=1
Epoch 4: acc=91.4%  → no improvement → counter=2
Epoch 5: acc=91.2%  → no improvement → counter=3 = patience!
                       → REDUCE LR: new_lr = old_lr × 0.5, counter resets
Epoch 6: acc=92.0%  → improved (with smaller LR)! best=92.0%, counter=0
```

Logic: "if accuracy hasn't improved for `patience` epochs straight, current LR is
probably too large to make further progress — cut it and try smaller steps."

Called after computing test accuracy each epoch:
```python
scheduler.step(test_acc)
```

### Early Stopping (Pure Logic, No Formula)

Tracks `best_accuracy_seen_so_far` and a `counter` of epochs since last real
improvement.

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_accuracy = 0
        self.best_model_weights = None
        self.should_stop = False

    def check(self, accuracy, model):
        if accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = accuracy
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model):
        model.load_state_dict(self.best_model_weights)
```

**Why `min_delta` matters:** without it, even a 0.001% "improvement" resets the
counter, so early stopping might never trigger even when the model is basically
flat-lined (just noise, not real progress). `min_delta=0.2` means only a genuine
≥0.2-point gain counts as improvement.

**Why we restore saved best weights, not just stop:**
```
Epoch 10: acc=94.5%  ← actual peak
Epoch 11: acc=94.2%  ← starting to slightly overfit
Epoch 12: acc=93.8%  ← getting worse
Epoch 13: acc=94.0%
Epoch 14: acc=93.9%  ← patience exhausted, STOP here

If we just kept the CURRENT (epoch 14) weights, we'd end up with a WORSE
model (93.9%) than what we actually achieved at epoch 10 (94.5%).
```

That's why every time a new best is found, we `deepcopy` and save the model's weights —
then at the end, restore that saved snapshot instead of keeping whatever the final
epoch left behind.

---

## Summary Table

| Technique | Core mechanism | Where applied | Train vs Eval difference |
|---|---|---|---|
| **BatchNorm** | `(x−μ)/√(σ²+ε)` then `γx̂+β` | After Conv/Linear, before activation | Train: batch stats. Eval: running average stats |
| **Dropout** | `x·mask / (1−p)` | After activation (post-ReLU) | Train: randomly zeroes neurons. Eval: fully active, no dropout |
| **Weight Decay** | `L_total = L + λΣw²` | Applied to optimizer, all weights | No train/eval difference — always active during training |
| **LR Scheduling** | `lr *= factor` when stuck | Wraps the optimizer | Only relevant during training |
| **Early Stopping** | Stop + restore best saved weights | Wraps the training loop | Only relevant during training |

---

## Key Bugs Encountered (Worth Remembering)

```
1. Duplicate attribute names silently overwrite each other
   self.bn1 = nn.BatchNorm2d(32)
   self.bn1 = nn.BatchNorm2d(64)   ← second line REPLACES the first, no error
                                       until forward() runs and shapes mismatch

2. BatchNorm1d size must exactly match the preceding Linear layer's output size
   self.fc1 = nn.Linear(64*7*7, 128)
   self.bn3 = nn.BatchNorm1d(256)   ← WRONG, must be 128 to match fc1's output

3. avg_loss / accuracy tracking placement matters
   Code indented inside the batch loop recalculates every batch instead of
   once per epoch — still "works" by accident (last value before loop end
   is correct) but pollutes any list you're appending to every batch.

4. Using CrossEntropyLoss but expecting BCELoss-style shapes
   CrossEntropyLoss expects raw logits (batch, num_classes) and integer
   class-index labels (batch,) — not one-hot encoded labels.
```

---

## Results (Fill in your own numbers here)

| Model | Train Acc | Test Acc | Gap |
|---|---|---|---|
| Baseline | 100.00% | 96.20% | 3.80% |
| + BatchNorm | 100.00% | 96.80% | 3.20% |
| + Dropout + Weight Decay | | | |
| + LR Scheduling + Early Stopping | | | |

**Observation:** MNIST is a visually simple dataset, so even with only 2000 training
samples, a CNN generalizes reasonably well — the overfitting gap here is modest (3–4%)
rather than dramatic. BatchNorm's main benefit shown here is training stability more
than large accuracy gains; Dropout and Weight Decay are expected to shrink the gap more
directly since they attack memorization head-on.

---

*Date: 2026*
*Status: Day 5 complete — Training techniques understood at both syntax and math level*
