import os
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
# Enable loading of truncated images to prevent errors on incomplete files
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CarlaWeatherDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir (str): Directory path containing weather simulation images.
            transform: Optional transformations to apply on the images.
        """
        self.data_dir = data_dir
        # List all .png files in the data directory
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

        # Use default transformations if none are provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        # Extract label (weather condition) from the filename
        # Create a mapping from weather name to numerical label
        self.label_mapping = {}
        for file in self.image_files:
            # Assuming file format "<weather_name>_<...>.png"
            weather_name = file.split('_')[0]
            if weather_name not in self.label_mapping:
                self.label_mapping[weather_name] = len(self.label_mapping)
        # Build the dataset list as tuples: (filename, numerical label)
        self.data = []
        for file in self.image_files:
            weather_name = file.split('_')[0]
            label = self.label_mapping[weather_name]
            self.data.append((file, label))
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]
        img_path = os.path.join(self.data_dir, file)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a blank black image if the image cannot be loaded
            image = Image.new('RGB', (256, 256), color=(0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloader(data_dir: str,
                   batch_size: int = 4,
                   num_workers: int = 2,
                   shuffle: bool = True,
                   pin_memory: bool = True,
                   persistent_workers: bool = True,
                   prefetch_factor: int = 2):
    """
    Create a DataLoader for the Carla weather dataset.

    Args:
        data_dir (str): Directory path containing the dataset images.
        batch_size (int): Number of images per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        shuffle (bool): Whether to shuffle the dataset.
        pin_memory (bool): Copy Tensors into CUDA pinned memory for faster GPU transfer.
        persistent_workers (bool): Keep workers alive between epochs if data loading is a bottleneck.
        prefetch_factor (int): Number of batches to prefetch for reducing I/O wait time.

    Returns:
        DataLoader: Configured PyTorch DataLoader instance.
    """
    dataset = CarlaWeatherDataset(data_dir)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers,
                            prefetch_factor=prefetch_factor)
    return dataloader


if __name__ == '__main__':
    # You can reuse this part to test the label correctness
    data_dir = os.path.join(os.path.dirname(__file__), 'carla_weather_dataset')
    dataset = CarlaWeatherDataset(data_dir)

    print("Label Mapping (Weather Name -> Label):")
    for weather_name, label in dataset.label_mapping.items():
        print(f"  {weather_name} -> {label}")

    print("\nChecking first 20 samples in the dataset:")
    for i in range(min(20, len(dataset))):
        file, label = dataset.data[i]
        weather_name = file.split('_')[0]
        expected_label = dataset.label_mapping[weather_name]
        print(f"File: {file} | Weather: {weather_name} | Assigned Label: {label} | Expected Label: {expected_label}")

        assert label == expected_label, f"Label mismatch for {file}"

