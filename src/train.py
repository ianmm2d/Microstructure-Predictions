# Training the CNN model

import torch
import matplotlib.pyplot as plt
from pipeline import DataPipeline
from model import CNN
import numpy as np

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs):
    """
    Train the model and evaluate on validation set after each epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
    
    Returns:
        train_losses: List of average training loss per epoch
        val_losses: List of average validation loss per epoch
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            targets = targets.view(-1,1)  # Flatten targets if necessary
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                targets = targets.view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses
torch.manual_seed(42)


#Data loading and preprocessing
image_dir = '../data/npy_images'
label_path = '../data/properties/material_property.csv'
batch_size = 32
test_split = 0.1
val_split = 0.1
pipeline = DataPipeline(image_dir, label_path,batch_size, test_split, val_split)
dataloaders = pipeline.load()
train_loader = dataloaders['train']
val_loader = dataloaders['val']

# Initialize the model, loss function, and optimizer
sample_input, _ = next(iter(train_loader))
_, c, h, w = sample_input.shape
model = CNN(input_channels=c, input_height=h, input_width=h)  # fix width → w
loss = torch.nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr)

# Train the model
train_losses, val_losses = train_and_evaluate(model, train_loader, val_loader, loss, optimizer, epochs=10)

# Plot training and validation loss curves
def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_loss_curves(train_losses, val_losses)


X_train_list = []
y_train_list = []
for batch_X, batch_y in train_loader:
    X_train_list.append(batch_X)
    y_train_list.append(batch_y)

X_train = torch.cat(X_train_list, dim=0)
y_train = torch.cat(y_train_list, dim=0)

model.eval()
with torch.no_grad():
    outputs = model(X_train[:10])
    print("Sample predictions:\n", outputs.squeeze())
    print("Corresponding targets:\n", y_train[:10].squeeze())

# Denormalize predictions and ground truth
with torch.no_grad():
    predictions = model(X_train.float()).squeeze().numpy()

y_true = y_train.numpy().squeeze()

# Apply inverse normalization using pipeline statistics
predictions = predictions * pipeline.y_std.item() + pipeline.y_mean.item()
y_true = y_true * pipeline.y_std.item() + pipeline.y_mean.item()

# Compute relative error
relative_errors = np.abs(predictions - y_true) / y_true
within_30_percent = relative_errors < 0.30
accuracy_30 = np.mean(within_30_percent)

print(f"Fraction within 30% error: {accuracy_30:.2%}")

# Plot histogram of relative errors
plt.figure(figsize=(8, 5))
plt.hist(relative_errors, bins=50, alpha=0.7)
plt.axvline(0.30, color='red', linestyle='--', label='30% Threshold')
plt.xlabel('Relative Error')
plt.ylabel('Count')
plt.title('Histogram of Relative Errors')
plt.legend()
plt.tight_layout()
plt.show()




