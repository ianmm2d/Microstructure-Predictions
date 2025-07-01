from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from .image_loader import get_image_data
from .targets_loader import get_y_values
import os

class DataPipeline:
    """
    DataPipeline class to load and preprocess image data and labels for 
    training, validation, and testing.
    It handles the loading of images from a directory and labels from a CSV file, 
    splits the data into training, validation, and test sets,
    and returns DataLoader objects for each set.
    """
    def __init__(
            self, 
            image_dir: str, 
            label_path: str, 
            batch_size=32, 
            test_split=0.1, 
            val_split=0.1
        ):
        self.image_dir = image_dir
        self.label_path = label_path
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split

    def load(self) -> dict[str, DataLoader]:
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label file not found: {self.label_path}")

        X, filenames = get_image_data(self.image_dir)
        y = get_y_values(self.label_path, filenames).unsqueeze(1)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, 
            y, 
            test_size=self.test_split, 
            random_state=42
        )
        
        val_ratio = self.val_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, 
            y_temp, 
            test_size=val_ratio, 
            random_state=42
        )

        #Normalize Lables
        self.y_mean = y_train.mean()
        self.y_std = y_train.std()

        y_train = ((y_train - self.y_mean) / self.y_std)
        y_val = ((y_val - self.y_mean) / self.y_std)
        y_test = ((y_test - self.y_mean) / self.y_std)

        return {
            'train': DataLoader(
                TensorDataset(X_train, y_train),    
                batch_size=self.batch_size, 
                shuffle=True
            ),
            'val': DataLoader(
                TensorDataset(X_val, y_val), 
                batch_size=self.batch_size
            ),
            'test': DataLoader(
                TensorDataset(X_test, y_test), 
                batch_size=self.batch_size
            )
        }
