import torch
import torchvision.transforms as transforms
from PIL import Image
import sys

from model import get_model


def predict(image_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(num_attributes=4)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # Same transforms as training (without augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int().cpu().numpy()[0]

    attributes_present = [
        f"Attr{i+1}" for i in range(4) if preds[i] == 1
    ]

    print("Attributes present:", attributes_present)


if __name__ == "__main__":
    image_path = sys.argv[1]
    predict(image_path)
