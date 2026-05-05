import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

trasnform= transforms.ToTensor()
test_dataset= torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         transform=trasnform,
                                         download=True)
train_dataset= torchvision.datasets.MNIST(root='./data',
                                         train=True,
                                         transform=trasnform,
                                         download=True)
print(len(train_dataset))
print(len(test_dataset))

image, label= train_dataset[0]
print(image.shape)
print(label)
print(type(label))

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle('MNIST Sample Images', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    # Get random image
    idx = np.random.randint(len(train_dataset))
    image, label = train_dataset[idx]
    
    # Convert to numpy and remove channel dimension
    image_np = image.squeeze().numpy()
    
    # Display
    ax.imshow(image_np, cmap='gray')
    ax.set_title(f'Label: {label}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_samples.png")
plt.show()

batch_size= 64
train_loader= DataLoader(train_dataset,
                         batch_size=batch_size,
                         shuffle=True)
test_loader= DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False)
print(len(train_loader),len(test_loader))

class MNIST_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(784,128)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(128,10)
    
    def forward(self,x):
        x= x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return(x)

model= MNIST_net()
print(model)
print(model.parameters)
for name, parameter in model.named_parameters():
    print(f"{name}:{parameter.shape} intially randomised")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

criterion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(),lr=0.001)
epochs=5
train_losses=[]
test_accuracies=[]
start_time = time.time()

for epoch in range(epochs):
    model.train()
    epoch_loss=0
    for batch_idx, (images, labels) in enumerate(train_loader):
        output= model(images)
        loss= criterion(output, labels) #loss is a tensor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item() #loss is converted into python number and then added to epoch loss
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}")
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs= model(images)
            _, predicted= torch.max(outputs,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            accuracy=100*correct/total
            test_accuracies.append(accuracy)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
                    f"Test Accuracy: {accuracy:.2f}%\n")

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

model.eval()
correct=0
total=0
predictions_all=[]
label_all=[]
with torch.no_grad():
    for images, labels in test_loader:
        output=model(images)
        _,predicted=torch.max(output,1)
        predictions_all.extend(predicted.numpy())
        label_all.extend(labels.numpy())
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
final_accuracy = 100 * correct / total
print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")
print(f"Correct: {correct}/{total}")
print(f"Incorrect: {total - correct}/{total}")
