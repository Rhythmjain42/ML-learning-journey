import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np

transform= transforms.ToTensor()
train_data= torchvision.datasets.MNIST(root='./data',
                                       train=True,
                                       transform=transform,
                                       download=True)
test_data= torchvision.datasets.MNIST(root='./data',
                                       train=False,
                                       transform=transform,
                                       download=True)
print(len(train_data))
print(len(test_data))

batch_size=64
train_loader= DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader= DataLoader(test_data,batch_size=batch_size,shuffle=False)

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1= nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2= nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.pool= nn.MaxPool2d(2,2)
        self.fc1= nn.Linear(64*7*7,128)
        self.fc2= nn.Linear(128,10)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        x= self.relu(self.conv1(x))
        x= self.pool(x)
        x= self.relu(self.conv2(x))
        x= self.pool(x)
        x= x.view(x.size(0),-1)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)

        return(x)
model= MNIST_CNN()
total_parameters= sum(p.numel() for p in model.parameters())
print(total_parameters)
print(model.parameters)

criterion = nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(), lr=0.001)
epochs= 3
train_loss=[]
test_accuracy=[]

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        output= model(images)
        loss= criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        if batch_idx%200==0:
            print(f"Epoch {epoch+1},batch {batch_idx}/{len(train_loader)}, loss: {loss.item()}")

        avg_loss=epoch_loss/len(train_loader)
        train_loss.append(avg_loss)

    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images, labels in test_loader:
            output= model(images)
            _,predicted= torch.max(output,1)
            total+=labels.size(0)
            correct+= (predicted==labels).sum().item()

    accuracy= 100*correct/total
    test_accuracy.append(accuracy)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

