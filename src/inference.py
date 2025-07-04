import torch
import numpy as np

def load_model(
        model: torch.nn.Module, 
        model_path: str, 
        device: torch.device
    ) -> torch.nn.Module:
    """
    Load a PyTorch model from a specified file path.
    
    Args:
        model (torch.nn.Module): The PyTorch model architecture to load weights into.
        model_path (str): Path to the saved model file.
        device (torch.device): Device to load the model onto (e.g., 'cpu' or 'cuda').
    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    model.to(device)
    # Load the state dictionary from the specified path
    state_dict = torch.load(model_path)
    # Load the weights into the model
    model.load_state_dict(state_dict)
    return model

def predict(
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        device: torch.device, 
        pipeline: object
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the provided model and dataloader.
    Args:
        model (torch.nn.Module): The PyTorch model to use for predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the input data.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        pipeline (object): Data pipeline object containing normalization parameters.
    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing predictions and targets.
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze()
            all_preds.append(outputs.cpu())
            all_targets.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Reshape and inverse transform
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    preds = preds * pipeline.y_std.item() + pipeline.y_mean.item()
    targets = targets * pipeline.y_std.item() + pipeline.y_mean.item()

    return preds, targets

def evaluate_predictions(
        preds: np.ndarray, 
        targets: np.ndarray, 
        threshold=0.30, 
        verbose=True
    ) -> tuple[float, float]:
    """
    Evaluate the predictions against the targets using relative error.
    Args:
        preds (np.ndarray): Array of predicted values.
        targets (np.ndarray): Array of target values.
        threshold (float): Threshold for relative error to consider as accurate.
        verbose (bool): Whether to print detailed evaluation information.
    Returns:
        tuple[float, float]: Mean relative error and accuracy of predictions.
    """
    relative_errors = np.abs(preds - targets) / targets
    within_threshold = relative_errors < threshold
    accuracy = np.mean(within_threshold)
    mean_error = np.mean(relative_errors)
    if verbose:
        # Printing random samples of predictions and targets
        sample_indices = np.random.choice(len(preds), size=10, replace=False)
        print("Sample predictions and targets:")
        print("Predictions:", preds[sample_indices])
        print("Targets:", targets[sample_indices])
        print(f"Percentage of data inside threshold accuracy: {accuracy*100:.2f}%")
        print(f"Percentage of Mean relative error for inference: {mean_error*100:.4f}%")
    return mean_error, accuracy
