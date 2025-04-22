import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import BeitFeatureExtractor as BeitImageProcessor, BeitConfig, BeitForImageClassification
from PIL import Image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import cv2

# === CONFIG ===
MODEL_DIR_CLEAN = "./beit-binary-skin"
MODEL_DIR_ADV = "./beit-skin-adv-mc"
TEST_DIR = "./test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8

# === LOAD MODEL FROM SAFETENSORS ===
def load_model(model_dir):
    config = BeitConfig.from_pretrained(model_dir)
    config.num_labels = 2
    model = BeitForImageClassification(config)
    state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()

# === LOAD PROCESSOR ===
def load_processor(model_dir):
    return BeitImageProcessor.from_pretrained(model_dir)

model_clean = load_model(MODEL_DIR_CLEAN)
model_adv = load_model(MODEL_DIR_ADV)
processor = load_processor(MODEL_DIR_CLEAN)

# === ATTACK FUNCTIONS ===
def fgsm_attack(image, label, epsilon, model):
    image.requires_grad = True
    output = model(pixel_values=image).logits
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    perturbed_image = image + epsilon * data_grad.sign()
    return torch.clamp(perturbed_image, 0, 1)

def pgd_attack(image, label, epsilon, alpha, steps, model):
    original = image.clone().detach()
    perturbed = image.clone().detach()
    for _ in range(steps):
        perturbed.requires_grad = True
        output = model(pixel_values=perturbed).logits
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        grad = perturbed.grad.data
        perturbed = perturbed + alpha * grad.sign()
        perturbation = torch.clamp(perturbed - original, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(original + perturbation, 0, 1).detach()
    return perturbed

# === LOAD TEST IMAGES ===
def load_test_batch():
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(TEST_DIR, transform=tf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for img, label in loader:
        return img.to(DEVICE), label.to(DEVICE), dataset.classes

# === METRIC EVALUATION ===
def evaluate_accuracy(images, labels, model):
    with torch.no_grad():
        logits = model(pixel_values=images).logits
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        return correct / len(labels), preds, logits.softmax(dim=-1)

# === GRAD-CAM ===
def get_gradcam_overlay(model, image, target_class):
    image = image.unsqueeze(0).to(DEVICE)
    image.requires_grad = True
    feature_maps = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    handle = model.beit.encoder.layer[-1].register_forward_hook(forward_hook)
    output = model(pixel_values=image).logits
    pred = output.argmax(dim=-1)
    loss = F.cross_entropy(output, target_class.unsqueeze(0))
    model.zero_grad()
    loss.backward()

    gradients = model.beit.encoder.layer[-1].output.dense.weight.grad
    pooled_gradients = gradients.mean(dim=1)[0]
    fmap = feature_maps[0][0].mean(0)
    cam = fmap.detach().cpu().numpy() * pooled_gradients.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    orig = np.uint8(255 * orig)
    overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
    handle.remove()
    return overlay

# === STREAMLIT UI ===
st.set_page_config(layout="wide")
st.title("🛡️ SkinGuardian: Adversarial Robustness Dashboard")

st.sidebar.header("⚙️ Attack Settings")
model_choice = st.sidebar.radio("Model", ["Clean-Trained", "Adversarial-Trained"])
attack_type = st.sidebar.selectbox("Attack Method", ["None", "FGSM", "PGD"])
epsilon = st.sidebar.slider("Perturbation Strength (ε)", 0.0, 0.2, 0.01, 0.005)

if attack_type == "PGD":
    alpha = st.sidebar.slider("Step Size (α)", 0.001, 0.02, 0.005, 0.001)
    steps = st.sidebar.slider("Number of Steps", 1, 20, 5, 1)
else:
    alpha, steps = None, None

if os.path.exists("robustness_metrics.csv"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Robustness Metrics")
    df = pd.read_csv("robustness_metrics.csv")
    st.sidebar.dataframe(df)

if st.button("Run Evaluation"):
    model = model_clean if model_choice == "Clean-Trained" else model_adv
    img, label, classes = load_test_batch()

    if attack_type == "None":
        adv_img = img
    elif attack_type == "FGSM":
        adv_img = fgsm_attack(img.clone(), label, epsilon, model)
    else:
        adv_img = pgd_attack(img.clone(), label, epsilon, alpha, steps, model)

    clean_acc, clean_preds, clean_probs = evaluate_accuracy(img, label, model)
    adv_acc, adv_preds, adv_probs = evaluate_accuracy(adv_img, label, model)

    st.subheader("📊 Accuracy Comparison")
    col1, col2 = st.columns(2)
    col1.metric("Clean Accuracy", f"{clean_acc * 100:.2f}%")
    col2.metric("Perturbed Accuracy", f"{adv_acc * 100:.2f}%")

    st.subheader("🔍 Predictions and Grad-CAM")
    for i in range(min(len(img), 4)):
        st.markdown(f"#### Image {i+1}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img[i].permute(1, 2, 0).cpu().numpy(), caption=f"Clean: {classes[clean_preds[i]]}")
        with col2:
            st.image(adv_img[i].permute(1, 2, 0).detach().cpu().numpy(), caption=f"Perturbed: {classes[adv_preds[i]]}")
        with col3:
            overlay = get_gradcam_overlay(model, img[i], label[i])
            st.image(overlay, caption="Grad-CAM Overlay")

    
    st.subheader("📈 Confidence Histograms")
    
    import plotly.graph_objects as go

    for class_idx, class_name in enumerate(classes):
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=clean_probs[:, class_idx].cpu().numpy(),
            name='Clean',
            opacity=0.75,
            marker_color='deepskyblue'
        ))
        fig.add_trace(go.Histogram(
            x=adv_probs[:, class_idx].cpu().numpy(),
            name='Perturbed',
            opacity=0.75,
            marker_color='tomato'
        ))
        fig.update_layout(
            title=f"Class: {class_name}",
            xaxis_title="Confidence",
            yaxis_title="Count",
            barmode='overlay',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            legend=dict(x=0.8, y=0.95)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Explanation
        st.markdown(f"**What this shows:** For the class `{class_name}`, this histogram compares the model's confidence in clean vs. adversarial settings. A sharp shift of 'Perturbed' confidence towards 0 suggests the model loses certainty under attack — a sign of vulnerability. If the Clean bars are higher at the right end (close to 1.0), it means the model was confident before perturbation.")


else:
    st.info("Click 'Run Evaluation' to visualize model performance.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, PyTorch, and Transformers")