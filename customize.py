import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()

        self.model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = True
       
        # 出力層（全結合層）を変更: 元は1000 → 任意のクラス数（ここでは10）
        self.model.fc = nn.Linear(in_features=512, out_features=10)

    def forward(self, input):
        return self.model(input)
    
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # Determine whether to use GPU (if available) or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Initialize running loss and correct predictions count for training
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the training data loader
        for inputs, labels in train_loader:
            # Move inputs and labels to the device (GPU or CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Reset the gradients to zero before the backward pass
            optimizer.zero_grad()

            # Forward pass: compute the model output
            outputs = model(inputs)
            # Get the predicted class (with the highest score)
            _, preds = torch.max(outputs, 1)
            # Compute the loss between the predictions and actual labels
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            loss.backward()
            # Perform the optimization step to update model parameters
            optimizer.step()

            # Accumulate the running loss and the number of correct predictions
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Compute average training loss and accuracy for this epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)

        # Set the model to evaluation mode for validation
        model.eval()
        # Initialize running loss and correct predictions count for validation
        running_loss = 0.0
        running_corrects = 0

        # Disable gradient computation for validation (saves memory and computations)
        with torch.no_grad():
            # Iterate over the validation data loader
            for inputs, labels in val_loader:
                # Move inputs and labels to the device (GPU or CPU)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass: compute the model output
                outputs = model(inputs)
                # Get the predicted class (with the highest score)
                _, preds = torch.max(outputs, 1)
                # Compute the loss between the predictions and actual labels
                loss = criterion(outputs, labels)

                # Accumulate the running loss and the number of correct predictions
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        # Compute average validation loss and accuracy for this epoch
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.float() / len(val_loader.dataset)

        # Print the results for the current epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

if __name__ == "__main__":
    model = CustomResNet18()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the train and validation datasets
    train_dataset = ImageFolder('custom_dataset/train', transform=transform)
    val_dataset = ImageFolder('custom_dataset/val', transform=transform)
    
    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Create data loaders for the train and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train(model,train_loader,val_loader,criterion,optimizer,num_epochs=1)

    torch.save(model.state_dict(), "model.pth")


