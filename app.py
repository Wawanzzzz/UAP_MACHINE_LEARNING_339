import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from cnn_model import SimpleCNN


# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Outfit Classification App",
    layout="centered"
)

st.title("👕 Klasifikasi Outfit")
st.write("Klasifikasi outfit **Biasa** vs **Skena** menggunakan Neural Network")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["biasa", "skena"]

# =========================
# TRANSFORM IMAGE
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model(model_name):
    if model_name == "CNN (Non-Pretrained)":
        model = SimpleCNN(num_classes=2)
        model.load_state_dict(torch.load("src/cnn_best.pth", map_location=device))

    elif model_name == "MobileNetV2 (Pretrained)":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        model.load_state_dict(torch.load("src/mobilenet_best.pth", map_location=device))

    else:
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        model.load_state_dict(torch.load("src/efficientnet_model.pth", map_location=device))

    model.to(device)
    model.eval()
    return model

# =========================
# UI INPUT
# =========================
model_choice = st.selectbox(
    "Pilih Model",
    ["CNN (Non-Pretrained)", "MobileNetV2 (Pretrained)", "EfficientNet-B0 (Pretrained)"]
)

uploaded_file = st.file_uploader(
    "Upload gambar outfit",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDIKSI
# =========================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input", use_column_width=True)

    model = load_model(model_choice)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, 1)

    st.success(f"🧠 Prediksi: **{class_names[pred_class.item()]}**")
    st.write(f"📊 Confidence: **{confidence.item()*100:.2f}%**")
