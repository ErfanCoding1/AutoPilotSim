import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

# Import the model module.
from weather_cnn import WeatherCNNClassifier

# Path to the folder containing all images.
DATASET_DIR_PATH = "C:/Users/erfan/PycharmProjects/CARLA_tutorial/src/assignement2_weather_classification/carla_weather_dataset"


# Custom dataset to load images from a single folder and extract labels from the filename.
class CustomWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory path with images.
            transform: Transformations to be applied on the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}

        # Iterate over all .png files in the folder.
        for filename in os.listdir(root_dir):
            if filename.endswith(".png"):
                label_str = filename.split('_')[0]
                if label_str not in self.label_map:
                    self.label_map[label_str] = len(self.label_map)
                self.image_paths.append(os.path.join(root_dir, filename))
                self.labels.append(self.label_map[label_str])

        # List of classes.
        self.classes = list(self.label_map.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Define image transformations.
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Use the custom dataset.
full_dataset = CustomWeatherDataset(root_dir=DATASET_DIR_PATH, transform=transform)
total_samples = len(full_dataset)
print(f"Total samples in dataset: {total_samples}")

# Split dataset: 80% training and 20% validation.
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

# Hyperparameters.
BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 10
INIT_LR = 1e-4
LR_PATIENCE = 5

# DataLoaders.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Directory to save model weights.
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

if __name__ == '__main__':
    # Select device: GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set number of classes based on the extracted labels.
    num_classes = len(full_dataset.classes)
    model = WeatherCNNClassifier(input_channels=3, num_classes=num_classes)
    model = model.to(device)

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=LR_PATIENCE)

    # Lists to store metrics for plotting.
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')

    # Training loop over epochs.
    for epoch in range(NUM_EPOCHS):
        # Training Phase.
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        total_train_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            epoch_train_correct += (preds == labels).sum().item()
            total_train_samples += inputs.size(0)

        avg_train_loss = epoch_train_loss / total_train_samples
        train_accuracy = epoch_train_correct / total_train_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation Phase.
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                epoch_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                epoch_val_correct += (preds == labels).sum().item()
                total_val_samples += inputs.size(0)

        avg_val_loss = epoch_val_loss / total_val_samples
        val_accuracy = epoch_val_correct / total_val_samples
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Print epoch summary.
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy * 100:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%")

        scheduler.step(avg_val_loss)

        # Save the best model weights based on validation loss.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(MODEL_SAVE_DIR, 'weather_classifier_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1}")

    # Save final model weights after completing all epochs.
    final_model_path = os.path.join(MODEL_SAVE_DIR, 'weather_classifier_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    # ----- Evaluate on Validation Set for Classification Report and Confusion Matrix ----- #
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # Generate the classification report.
    cls_report = classification_report(all_targets, all_preds, target_names=full_dataset.classes)
    print("\nClassification Report:")
    print(cls_report)

    # Save classification report to a text file.
    report_path = os.path.join(MODEL_SAVE_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(cls_report)
    print(f"Classification report saved to {report_path}")

    # Generate the confusion matrix.
    cm = confusion_matrix(all_targets, all_preds)

    # Plot confusion matrix using seaborn.
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(MODEL_SAVE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_path}")

    # ----- Plot Loss Curves ----- #
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    loss_curve_path = os.path.join(MODEL_SAVE_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Loss curve saved to {loss_curve_path}")

    # ----- Plot Accuracy Curves ----- #
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label="Validation Accuracy", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid()
    acc_curve_path = os.path.join(MODEL_SAVE_DIR, "accuracy_curve.png")
    plt.savefig(acc_curve_path)
    plt.close()
    print(f"Accuracy curve saved to {acc_curve_path}")
