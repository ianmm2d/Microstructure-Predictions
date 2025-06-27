# Training the CNN model

import torch
import matplotlib.pyplot as plt
from pipeline import DataPipeline
from model import CNN

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


# Plot training and validation losses
X_train_list = []
y_train_list = []
for batch_X, batch_y in train_loader:
    X_train_list.append(batch_X)
    y_train_list.append(batch_y)

X_train = torch.cat(X_train_list, dim=0)
y_train = torch.cat(y_train_list, dim=0)


def plot_results(model, X, y,y_mean, y_std):
    model.eval()
    with torch.no_grad():
        predictions = model(X.float()).squeeze().numpy()
    y_np = y.numpy().squeeze()

    predictions = predictions * y_std.item() + y_mean.item()
    y_np = y_np * y_std.item() + y_mean.item()

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_np)), y_np, label='Targets', alpha=0.7)
    plt.scatter(range(len(predictions)), predictions, label='Predictions', alpha=0.7)
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Young’s Modulus')
    plt.title('Predicted vs Actual Young’s Modulus')
    plt.show()

model.eval()
with torch.no_grad():
    outputs = model(X_train[:10])
    print("Sample predictions:\n", outputs.squeeze())
    print("Corresponding targets:\n", y_train[:10].squeeze())
plot_results(model, X_train, y_train, pipeline.y_mean, pipeline.y_std)

