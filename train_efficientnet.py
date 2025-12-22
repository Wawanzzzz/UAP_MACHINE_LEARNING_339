import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from src.dataloader import get_dataloaders

def main():
    train_loader, val_loader, _, class_names = get_dataloaders("src/data_split")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load EfficientNet-B0 pretrained
    model = models.efficientnet_b0(pretrained=True)

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Ganti classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "src/efficientnet_model.pth")
    print("🎉 Training EfficientNet selesai!")

if __name__ == "__main__":
    main()
