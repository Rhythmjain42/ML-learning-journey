import numpy as np
import matplotlib.pyplot as plt

X = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([0,0,0,0,0,0,1,1,1,1,1]).reshape(-1, 1) 
np.random.seed(42)
W=np.random.randn(1,1)*0.1
b=np.zeros((1,1))
print(W,b)
def sigmoid(z):
    return(1/(1+np.exp(-z)))
  
def forward(x,w,b):
    #multiply weights with input
    #find sigmoid to give answers on 0-1
    z=x@w+b
    return(sigmoid(z))
y_pred=forward(X,W,b)
print(y_pred)
def compute_loss(y_true,y_pred):
    #binary cross entropy loss function
    epsilon=1e-7
    y_pred= np.clip(y_pred,epsilon,1-epsilon)
    loss=-np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    return loss

initial_loss= compute_loss(Y,y_pred)
print(initial_loss)
def compute_gradient(X,y_true,y_pred):
    m= X.shape[0]
    error= y_pred-y_true
    dW= (X.T@error)/m
    db=np.mean(error)
    return dW,db
  
dW,db= compute_gradient(X,Y,y_pred)
print(dW,db)
def update_weights(w,b,dw,db,learning_rate):
    w= w-learning_rate*dw
    b= b-learning_rate*db
    return w,b
lr= 0.5
W_new,b_new=update_weights(W,b,dW,db,lr)
print(W_new,b_new)

y_pred_new=forward(X,W_new,b_new)
new_loss=compute_loss(Y,y_pred_new)
print(new_loss)

def train(X,Y,epochs=1000,lr=0.01):
    W=np.random.randn(1,1)*0.1
    b=np.zeros((1,1))
    losses=[]
    for epoch in range(epochs):
        y_pred= forward(X,W,b)
        loss=compute_loss(Y,y_pred)
        losses.append(loss)
        dW,db= compute_gradient(X,Y,y_pred)
        W,b=update_weights(W,b,dW,db,lr)
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d}: Loss = {loss:.4f}")
    return(W,b,losses)

W_final, b_final,final_loss=train(X,Y,1000,0.5)
y_pred_final = forward(X, W_final, b_final)
print("Input | True | Predicted")
print("-" * 30)

for i in range(len(X)):
    print(f"  {X[i,0]:.0f}   |  {Y[i,0]:.0f}   |   {y_pred_final[i,0]:.3f}")

print()
print("Final W:", W_final)
print("Final b:", b_final)
print("Final loss:", final_loss[-1])
plt.figure(figsize=(10, 5))
plt.plot(final_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True)

decision_boundary = -b_final / W_final
print(f"\nDecision Boundary:")
print(f"  W = {W_final[0,0]:.4f}")
print(f"  b = {b_final[0,0]:.4f}")
print(f"  Boundary at X = {decision_boundary[0,0]:.2f}")
print(f"  (Model predicts 0 when X < {decision_boundary[0,0]:.2f})")
print(f"  (Model predicts 1 when X > {decision_boundary[0,0]:.2f})")

# Visualize
# Create smooth prediction line
X_smooth = np.linspace(0, 11, 200).reshape(-1, 1)
y_smooth = forward(X_smooth, W_final, b_final)

plt.figure(figsize=(12, 6))

# Plot 1: Data points and decision boundary
plt.subplot(1, 2, 1)
plt.scatter(X[Y==0], Y[Y==0], c='blue', s=150, label='True: 0 (X≤5)', 
            marker='o', edgecolors='black', linewidth=2)
plt.scatter(X[Y==1], Y[Y==1], c='red', s=150, label='True: 1 (X>5)', 
            marker='s', edgecolors='black', linewidth=2)
plt.plot(X_smooth, y_smooth, 'green', linewidth=3, label='Model sigmoid output')
plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Decision threshold (0.5)')
plt.axvline(x=decision_boundary[0,0], color='orange', linestyle='--', 
            linewidth=2, label=f'Learned boundary (X={decision_boundary[0,0]:.2f})')
plt.xlabel('Input Value (X)', fontsize=12)
plt.ylabel('Prediction / Label', fontsize=12)
plt.title('Neural Network Decision Boundary', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)

# Plot 2: Show predictions for each point
plt.subplot(1, 2, 2)
y_pred_final = forward(X, W_final, b_final)
x_pos = np.arange(len(X))
width = 0.35

plt.bar(x_pos - width/2, Y.flatten(), width, label='True Label', 
        color='lightblue', edgecolor='black', linewidth=1.5)
plt.bar(x_pos + width/2, y_pred_final.flatten(), width, label='Prediction', 
        color='lightcoral', edgecolor='black', linewidth=1.5)
plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Input Value', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('True vs Predicted (Bar Chart)', fontsize=14, fontweight='bold')
plt.xticks(x_pos, X.flatten())
plt.legend()
plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('decision_boundary_visualization.png', dpi=150)
plt.show()

# Show numerical analysis
print("\nDetailed Predictions:")
print("Input | True | Predicted | Distance from 0.5 | Classification")
print("-" * 70)
for i in range(len(X)):
    pred = y_pred_final[i, 0]
    true = Y[i, 0]
    dist = abs(pred - 0.5)
    classification = "Class 1" if pred > 0.5 else "Class 0"
    confidence = "High" if dist > 0.3 else ("Medium" if dist > 0.1 else "Low")
    print(f"  {X[i,0]:2.0f}  |  {true:.0f}   |   {pred:.4f}   |     {dist:.4f}        | {classification} ({confidence} confidence)")






























