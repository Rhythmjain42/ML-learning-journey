# CNNs - Convolutional Neural Networks (Day 4)

## Overview

Built a CNN for MNIST and achieved 98-99% accuracy - better than the fully connected network from Day 3 (97%).

**Key insight: CNNs preserve spatial structure of images, making them more efficient and accurate for computer vision.**

---

## What Changed from Day 3

### Day 3: Fully Connected Network
Architecture: 784 → 128 → 10

Treats image as flat list (ignores spatial structure)
Every pixel connects to every neuron
~100,000 parameters
Accuracy: 97%


### Day 4: Convolutional Network
Architecture: Input → Conv → Pool → Conv → Pool → FC → Output

Preserves 2D spatial structure
Learns local patterns (edges, shapes)
~50,000 parameters (more efficient!)
Accuracy: 98-99%


---

## What I Learned

### 1. Convolution - The Core Concept

**Think of it like sliding a magnifying glass:**
Image: 28×28 pixels
Filter: 3×3 "magnifying glass"
Process: Slide filter across image
Result: Detect patterns (edges, corners, shapes)

**Example: Edge Detection**
```python
# 5×5 image with vertical edge
image = [
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
]

# Vertical edge detection filter
filter = [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
]

# Convolution detects the edge at x=2
# Output will have high values where edge exists
```

**In CNNs:**
- Filters are LEARNED, not hand-coded
- Network learns what patterns to detect
- Early layers: edges, corners
- Deep layers: complex shapes, objects

---

### 2. Key CNN Concepts

#### Convolution Layer
```python
nn.Conv2d(in_channels, out_channels, kernel_size, padding)

# Example:
nn.Conv2d(1, 32, kernel_size=3, padding=1)
# Input: 1 channel (grayscale)
# Output: 32 channels (32 different filters)
# Filter size: 3×3
# Padding: 1 (keeps size same)
```

**What happens:**
- Input: (batch, 1, 28, 28)
- Apply 32 different 3×3 filters
- Output: (batch, 32, 28, 28)
- Each of 32 channels detects different pattern

---

#### Pooling Layer
```python
nn.MaxPool2d(kernel_size, stride)

# Example:
nn.MaxPool2d(2, 2)
# Takes 2×2 region, keeps maximum value
# Reduces size by half
```

**Why pooling?**
- Reduces computation (smaller size)
- Makes features more robust (slight position changes don't matter)
- Focuses on "is pattern present?" not "exactly where?"

**Example:**
Input: 28×28
After MaxPool(2,2): 14×14  (half the size)
After another MaxPool(2,2): 7×7

---

#### Filter/Kernel Size
```python
kernel_size=3  # 3×3 filter (most common)
kernel_size=5  # 5×5 filter (larger receptive field)
kernel_size=1  # 1×1 filter (channel mixing)
```

---

#### Stride
```python
stride=1  # Move filter 1 pixel at a time (default)
stride=2  # Move filter 2 pixels (reduces output size)
```

---

#### Padding
```python
padding=0  # No padding (output shrinks)
padding=1  # Add 1 pixel border (keeps size same for 3×3 filter)
```

**Effect on output size:**
Input: 28×28
Filter: 3×3, stride=1, padding=0 → Output: 26×26 (shrinks)
Filter: 3×3, stride=1, padding=1 → Output: 28×28 (same!)

---

### 3. CNN Architecture

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Block 1: Conv → ReLU → Pool
        x = self.relu(self.conv1(x))  # 28×28×32
        x = self.pool(x)               # 14×14×32
        
        # Block 2: Conv → ReLU → Pool
        x = self.relu(self.conv2(x))  # 14×14×64
        x = self.pool(x)               # 7×7×64
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)     # 3136
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
```

**Step-by-step dimensions:**
Input:    (batch, 1, 28, 28)
Conv1:    (batch, 32, 28, 28)  [32 filters applied]
Pool1:    (batch, 32, 14, 14)  [reduced by half]
Conv2:    (batch, 64, 14, 14)  [64 filters applied]
Pool2:    (batch, 64, 7, 7)    [reduced by half]
Flatten:  (batch, 3136)        [7×7×64 = 3136]
FC1:      (batch, 128)
FC2:      (batch, 10)          [final output]

---

### 4. Why ReLU Instead of Sigmoid?

```python
# Day 1-3: Used Sigmoid
sigmoid(x) = 1 / (1 + e^(-x))
# Output: 0 to 1
# Problem: Vanishing gradients in deep networks

# Day 4: Use ReLU
ReLU(x) = max(0, x)
# Output: 0 to infinity
# Benefits:
#   - Faster training
#   - No vanishing gradients
#   - Introduces non-linearity
#   - Computationally efficient
```

**When to use:**
- ReLU: Hidden layers (almost always)
- Sigmoid: Binary classification output
- Softmax: Multi-class output (inside CrossEntropyLoss)
- None: Final layer with CrossEntropyLoss (it includes softmax)

---

### 5. Loss Functions - Important!

**Common mistake I made:**
```python
# WRONG for multi-class:
criterion = nn.BCELoss()  # For BINARY classification only!

# RIGHT for multi-class:
criterion = nn.CrossEntropyLoss()  # For 10 classes (digits 0-9)
```

**When to use which:**

| Task | Loss Function | Output Layer |
|------|---------------|--------------|
| Binary (2 classes) | BCELoss or BCEWithLogitsLoss | 1 neuron + Sigmoid |
| Multi-class (3+ classes, one correct) | CrossEntropyLoss | N neurons, no activation |
| Multi-label (multiple correct) | BCEWithLogitsLoss | N neurons, no activation |
| Regression | MSELoss or L1Loss | N neurons, no activation |

**For MNIST:**
- 10 classes (digits 0-9)
- Only ONE correct answer
- Use: `CrossEntropyLoss`

**CrossEntropyLoss expects:**
```python
outputs = model(images)  # Shape: (64, 10) - raw scores (logits)
labels = ...             # Shape: (64,) - class indices [0, 1, 2, ..., 9]

loss = criterion(outputs, labels)  # ✓ Correct!
```

**Common error:**
```python
# If you get: "target size different from input size"
# Check:
print(outputs.shape)  # Should be (batch_size, num_classes)
print(labels.shape)   # Should be (batch_size,)

# If labels are (batch_size, num_classes), you're using wrong loss!
```

---

## Results Comparison

| Metric | Day 3 (FC) | Day 4 (CNN) |
|--------|-----------|-------------|
| Architecture | 784→128→10 | Conv→Pool→Conv→Pool→FC |
| Parameters | ~100,000 | ~50,000 |
| Accuracy | 97.5% | 98-99% |
| Training time | ~5 min | ~3 min |
| Efficiency | Treats as flat list | Preserves spatial structure |

**CNNs are both more efficient AND more accurate!**

---

## What Makes CNNs Better for Images?

### Fully Connected Problems:
Image: 28×28 = 784 pixels
FC Layer: 784 → 128
Issues:

Ignores that nearby pixels are related
Top-left and bottom-right pixels treated equally
No concept of "edges" or "shapes"
Too many parameters


### CNN Advantages:
Uses 3×3 filters sliding across image
Benefits:

Nearby pixels processed together
Detects local patterns (edges, corners)
Shares weights across image (fewer parameters)
Position-invariant (detects pattern anywhere)


**Example:**
- FC network: Learns "pixel 42 and pixel 315 together mean digit 3"
- CNN: Learns "vertical edge + horizontal edge + curve = digit 3"

The CNN approach is more generalizable and efficient!
