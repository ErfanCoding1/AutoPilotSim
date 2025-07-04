import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

# Import your trained weather CNN classifier model.
from weather_cnn import WeatherCNNClassifier

# Define the class names corresponding to the weather conditions.
class_names = ['ClearNight', 'ClearNoon', 'Foggy', 'SoftRainNoon', 'WetCloudyNoon']

# Path to the saved model weights (modify as needed).
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_model', 'weather_classifier_final.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the trained model.
model = WeatherCNNClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define image preprocessing (same as used during training).
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define directory to save the predicted plots/images.
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'predicted_results')
os.makedirs(SAVE_DIR, exist_ok=True)

def predict(image_path: str, ground_truth=None):
    """
    Predict the weather condition of an image, display the result,
    and save the image with overlaid prediction and ground truth text.
    :param image_path: Full path to the image file.
    :param ground_truth: Optional ground truth label (for display purposes).
    """
    # Read image using OpenCV and keep a copy for visualization.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image from {image_path}")
        return

    original_image = image.copy()
    # Convert BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess the image.
    image = transform(Image.fromarray(image))
    image = image.unsqueeze(0).to(device)  # add batch dimension

    # Inference: obtain model output.
    with torch.no_grad():
        output = model(image)
        # For multi-class classification, get the class with maximum logit.
        _, predicted_idx = torch.max(output, 1)

    predicted_class = class_names[predicted_idx.item()]
    gt_text = f"GT: {ground_truth}" if ground_truth is not None else "GT: Unknown"
    pred_text = f"Prediction: {predicted_class}"

    # Add text labels on the image for visualization.
    cv2.putText(original_image, gt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(original_image, pred_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the image.
    cv2.imshow("Weather Classification", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the resulting image.
    save_filename = os.path.basename(image_path)
    save_path = os.path.join(SAVE_DIR, save_filename)
    cv2.imwrite(save_path, original_image)
    print(f"Saved predicted image to: {save_path}")


# Example usage: walking through the test directory and predicting each image.
if __name__ == "__main__":
    # Modify the path according to the location of your test dataset.
    data_dir = os.path.join(os.path.dirname(__file__), 'carla_weather_dataset')

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path_ = os.path.join(root, file)
                # Optionally, assume ground truth is stored in the name of the parent folder.
                ground_truth_ = os.path.basename(root)
                predict(image_path_, ground_truth_)
