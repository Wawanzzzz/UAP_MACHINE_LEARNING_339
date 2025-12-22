import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataloader import get_dataloaders
from src.cnn_model import SimpleCNN

# ==================================================
# KONFIGURASI
# ==================================================
DATA_DIR = "src/data_split"
BATCH_SIZE = 32
MODEL_PATH = "src/cnn_best.pth"

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
model = SimpleCNN(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
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
print("\n=== Classification Report ===")
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
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - CNN Non-Pretrained")
plt.tight_layout()
plt.show()
