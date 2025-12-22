import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report
from src.dataloader import get_dataloaders

def main():
    _, _, test_loader, class_names = get_dataloaders("src/data_split")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("src/efficientnet_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n=== Classification Report (EfficientNet) ===")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    main()
