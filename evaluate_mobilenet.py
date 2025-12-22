import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
import torch.nn as nn

from src.dataloader import get_dataloaders

# ==================================================
# KONFIGURASI
# ==================================================
DATA_DIR = "src/data_split"
BATCH_SIZE = 32
MODEL_PATH = "src/mobilenet_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================
# LOAD DATA
# ==================================================
_, _, test_loader, class_names = get_dataloaders(
    DATA_DIR, batch_size=BATCH_SIZE
)

# ==================================================
# LOAD MODEL
# ==================================================
model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==================================================
# PREDIKSI
# ==================================================
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

# ==================================================
# CLASSIFICATION REPORT
# ==================================================
print("\n=== Classification Report (MobileNet) ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# ==================================================
# CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - MobileNetV2")
plt.tight_layout()
plt.show()
