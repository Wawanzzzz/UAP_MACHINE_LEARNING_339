import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from src.dataloader import get_dataloaders

# ==================================================
# KONFIGURASI
# ==================================================
DATA_DIR = "src/data_split"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================
# LOAD DATA
# ==================================================
train_loader, val_loader, _, class_names = get_dataloaders(
    DATA_DIR, batch_size=BATCH_SIZE
)

# ==================================================
# LOAD PRETRAINED MODEL
# ==================================================
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Ganti classifier
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model = model.to(DEVICE)

# ==================================================
# LOSS & OPTIMIZER
# ==================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# ==================================================
# TRAINING
# ==================================================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    model.train()
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # ------------------------------
    # VALIDATION
    # ------------------------------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "src/mobilenet_best.pth")
        print("✅ Model terbaik disimpan")

print("\n🎉 Training MobileNet selesai!")
