# Training the CNN model
import torch
import matplotlib.pyplot as plt

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs, device=None, seed=42):
    """
    Train the model and evaluate on validation set after each epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        device: torch.device object (e.g., torch.device("cuda") or torch.device("cpu"))
        seed: Int value for torch seed
    
    Returns:
        train_losses: List of average training loss per epoch
        val_losses: List of average validation loss per epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Defining seed for reproducibility
    torch.manual_seed(seed)
    model = model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
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
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses


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