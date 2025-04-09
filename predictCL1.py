import torch
from PIL import Image
import torchvision.transforms as transforms
from modelCNN import CNN
import os

def predict(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    return "Healthy" if predicted.item() == 1 else "Diseased"

if __name__ == "__main__":
    model_path = "trained_model_client_1.pt"  
    image_folder = "data/valid"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    print("Predictions:")
    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(image_folder, filename)
            label = predict(path, model, transform)
            print(f"{filename} → {label}")
