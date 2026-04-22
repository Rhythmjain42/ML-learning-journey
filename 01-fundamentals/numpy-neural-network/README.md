1.	Np.reshape- In NumPy, array.reshape(-1, 1) is used to transform an array into a 2D column vector
-1 (Unknown Dimension): Tells NumPy to automatically calculate the number of rows based on the total number of elements in the original array. Only one dimension can be set to -1.
1 (Fixed Dimension): Explicitly sets the number of columns to 1.
Format of passing dimension is row X column.
2.	In NumPy, .T is an attribute of the ndarray object that returns a view of the transposed array. It is a concise shorthand for calling the numpy.transpose() function without arguments. 
Key Features of .T
Axis Reversal: It reverses the order of the dimensions. For a 2D matrix, it flips rows and columns: an element at index (i, j) moves to (j, i).
View, Not Copy: It does not create a new array in memory but provides a different "view" of the existing data using different strides. Modifications to the original array will reflect in the .T view and vice-versa.
3.	np.random.randn(1,1) generates a NumPy array (a 2D matrix) containing a single random float sampled from a univariate "normal" (Gaussian) distribution with a mean of 0 and a variance of 1. The output typically ranges from negative to positive infinity, clustered around zero
4.	here first we make dataset
5.	then initialize sigmoid functions which are needed to convert output in 0-1 range for getting output as probability
6.	in forward, we multiply weights with input and add bias to it, once we get result of it we apply sigmoid in result and return it
7.	then after a single forward pass we compute loss, loss is calculated using BCE or MSE loss functions
8.	once loss is computed then we compute gradients for change in weights and bias [y_pred-y_true]
9.	we update weights by multiplying learning rate with computed gradients
Derivative of Cross-Entropy Loss:
Loss = -[y*log(p) + (1-y)*log(1-p)]
where p = sigmoid(z) = 1/(1 + e^(-z))
Taking derivative:
∂Loss/∂p = -[y/p - (1-y)/(1-p)]
But we need ∂Loss/∂z (where z = X*W + b)
Using chain rule:
∂Loss/∂z = ∂Loss/∂p * ∂p/∂z
∂p/∂z = p(1-p)  (derivative of sigmoid)
So:
∂Loss/∂z = -[y/p - (1-y)/(1-p)] * p(1-p)
         = -[y(1-p) - (1-y)p]
         = -[y - yp - p + yp]
         = -[y - p]
         = p - y
         = y_pred - y_true
Therefore: error = y_pred - y_true ✓
Loss L = -[y*log(σ(z)) + (1-y)*log(1-σ(z))]
where σ(z) = 1/(1 + e^(-z)), z = W*x + b
Step 1: ∂L/∂σ
∂L/∂σ = -[y/σ - (1-y)/(1-σ)]
Step 2: ∂σ/∂z (sigmoid derivative)
∂σ/∂z = σ(1-σ)
Step 3: Chain rule
∂L/∂z = ∂L/∂σ * ∂σ/∂z
      = -[y/σ - (1-y)/(1-σ)] * σ(1-σ)
      = -[y(1-σ) - (1-y)σ]
      = -[y - yσ - σ + yσ]
      = -(y - σ)
      = σ - y
      = y_pred - y_true
For MSE + Sigmoid
Loss L = (y - σ(z))²
Step 1: ∂L/∂σ
∂L/∂σ = 2(σ - y)
Step 2: ∂σ/∂z
∂σ/∂z = σ(1-σ)
Step 3: Chain rule
∂L/∂z = ∂L/∂σ * ∂σ/∂z
      = 2(σ - y) * σ(1-σ)
      = 2(y_pred - y_true) * y_pred * (1 - y_pred)
NOT as simple! The σ(1-σ) term stays.

