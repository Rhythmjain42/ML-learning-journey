# Day 6: Loss Functions Survey

Covers classification losses (BCE, CrossEntropy, Focal), regression losses (MSE, MAE,
Huber), and CTC Loss (critical for future OCR/Devanagari work) — with math derivations,
not just usage syntax.

---

## 1. Binary Cross Entropy (BCE)

### Where the Formula Comes From

BCE comes from **Maximum Likelihood Estimation** — we want model parameters that make
the true labels as probable as possible under the model's predictions.

```
For one prediction: y_true ∈ {0,1}, y_pred = p (predicted probability of "yes")

Likelihood combining both cases:
  L = p^(y_true) × (1-p)^(1-y_true)

Take log (turns product into sum, avoids numerical underflow):
  log(L) = y_true·log(p) + (1-y_true)·log(1-p)

We want to MAXIMIZE this, but optimizers minimize loss, so negate it:

  BCE Loss = -[y_true·log(p) + (1-y_true)·log(1-p)]
```

### Worked Example

```
Confident + correct:  y=1, p=0.9
  Loss = -log(0.9) = 0.105   (small)

Confident + WRONG:    y=1, p=0.1
  Loss = -log(0.1) = 2.303   (much larger)
```

**Why the asymmetry:** `log(x)` → −∞ as `x→0`, but flattens near `x=1`. Confidently
wrong predictions get punished far more harshly than confidently correct predictions
get rewarded — this pushes strong gradient signal exactly where the model is making
overconfident mistakes.

### Where Used

- Binary classification (1 neuron, Sigmoid, output in (0,1))
- Multi-label classification (N neurons, each an independent Sigmoid yes/no)
- **Not** for multi-class-single-answer problems (that's CrossEntropy)

**Practical note:** use `BCEWithLogitsLoss` over `BCELoss` + separate `Sigmoid` — it
combines both internally for better numerical stability.

---

## 2. CrossEntropy Loss

### Extending BCE to N Classes

```
General form (N classes, one-hot y):
  Loss = -Σ(y_i · log(p_i))   for i=1..N

Since y_i = 0 for every class except the true class t:
  Loss = -log(p_t)

Just: negative log of the probability assigned to the correct class.
```

### Where `p_i` Comes From — Softmax

Models output raw logits (any real number). Softmax converts them to valid
probabilities:

```
p_i = e^(z_i) / Σ(e^(z_j))   for j=1..N
```

**Why `e^x` specifically:** always positive (satisfies probability constraint),
monotonically increasing (preserves relative confidence ordering), and dividing by the
sum forces everything to sum to 1.

### Worked Example

```
Logits: [2.0, 0.5, 0.3, 5.0, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1], true class = index 3

Exponentiate: 7.39, 1.65, 1.35, 148.41, 1.11, 1.22, 1.11, 1.35, 1.22, 1.11
Sum = 165.92
p_3 = 148.41 / 165.92 = 0.8945

Loss = -log(0.8945) = 0.1115   (small — model's highest score IS the true class)
```

If the highest logit were at the WRONG index instead, p_3 would shrink drastically
(e.g. to ~0.02), giving `Loss = -log(0.02) ≈ 3.9` — same asymmetric explosion as BCE.

### Numerical Stability Note

```python
# Don't do this manually:
outputs = softmax(model(x))
loss = nn.NLLLoss()(log(outputs), labels)   # risky: log(near-zero) issues

# Do this instead:
outputs = model(x)                           # raw logits, NO activation
loss = nn.CrossEntropyLoss()(outputs, labels)  # softmax + log combined internally, stable
```

This is why output layers feeding into `CrossEntropyLoss` should have **no activation
function** — the loss expects raw logits.

### Where Used

- Multi-class, exactly ONE correct answer (MNIST digits, future Devanagari character
  classification)
- **Not** for multi-label problems

---

## 3. Focal Loss

### The Problem It Solves

Standard CrossEntropy weighs every sample equally. With severe class imbalance (or lots
of "easy" background examples, as in object detection), the abundant easy examples
drown out the gradient signal from rare/hard examples.

### The Formula

```
FL(p_t) = -(1 - p_t)^γ · log(p_t)

p_t = model's predicted probability for the TRUE class
γ (gamma) = focusing parameter, typically 2
```

**Mechanism:**
```
p_t HIGH (already confident + correct) → (1-p_t)^γ becomes very small
                                        → loss contribution shrinks toward 0

p_t LOW (wrong or unsure)              → (1-p_t)^γ stays close to 1
                                        → loss barely reduced from plain CrossEntropy
```

**Result:** easy, already-correct examples contribute almost nothing to training. Hard,
rare, or wrong examples dominate the gradient — exactly what's needed when a few rare
classes are getting ignored by standard CrossEntropy.

### Where Used

- Object detection (huge background/foreground imbalance)
- Rare-class classification problems
- Likely relevant for rare Devanagari conjuncts/ligatures later

---

## 4. Regression Losses: MSE, MAE, Huber

| Loss | Formula | Growth | Use when |
|---|---|---|---|
| MSE | `(pred-true)²` | Quadratic | Outliers are meaningful, want to punish large errors hard |
| MAE | `\|pred-true\|` | Linear | Outliers are likely noise, want robust/stable training |
| Huber | MSE near 0, MAE far from 0 (switch at `delta`) | Hybrid | Want MSE's smoothness + MAE's outlier robustness |

MSE grows quadratically — an error of 10 is punished 100× more than an error of 1 (not
10×). MAE grows linearly — proportional punishment regardless of error size. Huber
gives the smooth gradient of MSE near the correct answer, but doesn't let a few extreme
outliers dominate training the way pure MSE would.

---

## 5. CTC Loss (Connectionist Temporal Classification)

### The Problem It Solves

In OCR (and speech recognition), you have a sequence input (image scanned left-to-right)
but don't know the exact alignment between input positions and output characters — how
many pixel-columns belong to each character, where one character ends and the next
begins.

### The Mechanism

The model outputs a prediction at every timestep. CTC introduces a special **blank**
token and a collapse rule:

```
Example — image of "cat", model predicts at 8 timesteps:

Timestep:   1   2   3   4   5   6   7   8
Predicted:  c   c   c   a   a   -   t   t

Collapse rule:
  1. Merge consecutive repeated characters: "cccaa-tt" → "ca-t"
  2. Remove blank tokens: "ca-t" → "cat"

Result: "cat" — without ever needing pixel-perfect character boundaries in training data.
```

### PyTorch Usage Pattern

```python
ctc_loss = nn.CTCLoss(blank=0)   # index 0 reserved for blank

log_probs = model_output.log_softmax(2)   # (timesteps, batch, num_classes)
targets = torch.tensor([1, 2, 3])          # true label sequence, no blanks needed
input_lengths = torch.tensor([T])          # timesteps produced
target_lengths = torch.tensor([3])         # true label length

loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
```

### Why This Matters for Future Devanagari Work

Manuscript line images have variable spacing, distortion, and connected characters
(Devanagari's horizontal headline connects characters). CTC allows training an
end-to-end image→text model without needing perfectly segmented individual characters
as training labels — the model learns the alignment implicitly.

---

## Decision Table

| Problem Type | Loss | Output Layer |
|---|---|---|
| Binary classification | BCEWithLogitsLoss | 1 neuron, no activation |
| Multi-class (one correct) | CrossEntropyLoss | N neurons, no activation |
| Multi-label (several correct) | BCEWithLogitsLoss | N neurons, no activation |
| Imbalanced classification | Focal Loss | N neurons, no activation |
| Standard regression | MSELoss | N neurons, no activation |
| Regression with outliers | HuberLoss / L1Loss | N neurons, no activation |
| Sequence recognition (OCR, speech) | CTCLoss | Per-timestep predictions |

---

*Date: 2026*
*Status: Day 6 complete — Loss function landscape covered at math + practical level*
