# app.py
import io, os, sys, uuid
from flask import Flask, request, render_template, url_for, redirect
from PIL import Image
import numpy as np

# ONNX-Runtime for fast inference & occlusion
import onnxruntime as ort

# ONNX → PyTorch converter
import onnx
from onnx2torch import convert

# PyTorch + Grad-CAM
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# LIME
from lime import lime_image
from skimage.segmentation import slic

# Matplotlib “Agg” backend (no GUI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (Optional) HuggingFace BEiT extractor for correct mean/std
from transformers import BeitFeatureExtractor

# ─── Flask setup ───────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ─── 1) Load ONNX for inference + occlusion ────────────────────────────────────
onnx_path = os.path.join(BASE_DIR, "beit_finetuned_model.onnx")
ort_sess  = ort.InferenceSession(onnx_path)

# ─── 2) Convert ONNX → PyTorch for Grad-CAM ───────────────────────────────────
onnx_model = onnx.load(onnx_path)
pt_model   = convert(onnx_model).eval().to(
                 torch.device("cuda" if torch.cuda.is_available() else "cpu")
             )

# (Optional) for exact BEiT normalization
feat_ext = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224")


# ─── 3) Preprocess for ONNX (resize + scale) ─────────────────────────────────
def preprocess_onnx(pil: Image.Image) -> np.ndarray:
    img = pil.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0   # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1))[None]       # [1,3,224,224]
    return arr


# ─── 4) ONNX inference + occlusion ───────────────────────────────────────────
def run_inference(arr: np.ndarray):
    inp_name = ort_sess.get_inputs()[0].name
    logits   = ort_sess.run(None, {inp_name: arr})[0]  # [1,2]
    exps     = np.exp(logits)
    probs    = exps / exps.sum(axis=1, keepdims=True)
    p_cancer = float(probs[0,1])
    label    = "Positive" if p_cancer>0.5 else "Negative"
    return label, p_cancer

def compute_occlusion(arr: np.ndarray) -> str:
    H, W      = arr.shape[2], arr.shape[3]
    baseline  = arr.mean()
    heatmap   = np.zeros((H, W), dtype=np.float32)
    # slide a 32×32 patch with stride=16
    for y in range(0, H, 16):
        for x in range(0, W, 16):
            tmp = arr.copy()
            tmp[0, :, y:y+32, x:x+32] = baseline
            _, p = run_inference(tmp)
            y1, x1 = min(y+32, H), min(x+32, W)
            heatmap[y:y1, x:x1] = p
    # normalize & overlay
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    orig    = arr[0].transpose(1, 2, 0)  # [H,W,3]
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(orig, alpha=0.8)
    ax.imshow(heatmap, cmap="jet", alpha=0.4)
    ax.axis("off")
    out = os.path.join("static","explain","occlusion.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return "explain/occlusion.png"



# ─── 5) Genuine Grad-CAM (PyTorch) ────────────────────────────────────────────
def compute_gradcam(arr: np.ndarray) -> str:
    device = next(pt_model.parameters()).device

    # a) reshape_transform for ViT token → spatial map
    def reshape_transform(tensor, height=14, width=14):
      x = (tensor[:,1:,:]
           .reshape(tensor.size(0), height, width, tensor.size(2))
           .permute(0,3,1,2))
      return x

    # b) wrap GraphModule → forward(x)→logits
    class Wrapper(torch.nn.Module):
      def __init__(self, gm): super().__init__(); self.gm = gm
      def forward(self, x):      return self.gm(x)

    wrapper = Wrapper(pt_model).to(device).eval()
    inp     = torch.from_numpy(arr).float().to(device)

    # c) pick predicted class
    with torch.no_grad():
      logits = wrapper(inp)
    pred_idx = int(logits.argmax(dim=1).item())
    targets  = [ClassifierOutputTarget(pred_idx)]

    # d) grab the last encoder layer
    layers = [m for n,m in pt_model.named_modules() if "encoder" in n and "layer" in n]
    target_layer = layers[-1]

    # e) run GradCAM
    cam = GradCAM(
      model=wrapper,
      target_layers=[target_layer],
      reshape_transform=reshape_transform
    )
    grayscale = cam(inp, targets=targets)[0]

    # f) overlay & save
    orig    = arr[0].transpose(1,2,0)
    overlay = show_cam_on_image(orig, grayscale, use_rgb=True)
    out     = os.path.join("static","explain","gradcam.png")
    plt.imsave(out, overlay)
    return "explain/gradcam.png"


# ─── 6) LIME Explainability ───────────────────────────────────────────────────
def compute_lime(pil: Image.Image) -> str:
    img_np    = np.array(pil.convert("RGB"))
    explainer = lime_image.LimeImageExplainer(random_state=42)

    def predict_fn(batch):
        out = []
        for patch in batch:
            arr, _ = preprocess_onnx(Image.fromarray(patch)), None
            _, p   = run_inference(arr)
            out.append([1-p, p])
        return np.array(out)

    exp = explainer.explain_instance(
        img_np, predict_fn,
        top_labels=1, hide_color=0,
        num_samples=300,
        segmentation_fn=lambda x: slic(x, n_segments=100, compactness=10, sigma=1)
    )

    mask = exp.get_image_and_mask(
        exp.top_labels[0],
        positive_only=True,
        num_features=3,
        hide_rest=True
    )[1] > 0

    overlay = img_np.astype(np.float32)/255.0
    red     = np.zeros_like(overlay); red[...,0]=1.0
    alpha   = 0.6
    overlay[mask]   = overlay[mask]*(1-alpha) + red[mask]*alpha
    overlay[~mask] *= 0.4

    out = os.path.join("static","explain","lime.png")
    Image.fromarray((overlay*255).astype(np.uint8)).save(out)
    return "explain/lime.png"





# ─── 7) Flask routes ───────────────────────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def index():
    if request.method=="POST" and (f := request.files.get("file")):
        pil = Image.open(io.BytesIO(f.read()))
        arr = preprocess_onnx(pil)

        label, prob = run_inference(arr)
        # save upload
        fn   = f"{uuid.uuid4().hex}.png"
        path = os.path.join(app.config["UPLOAD_FOLDER"], fn)
        pil.save(path)
        if label != "Positive":
            prob = 1 - prob

        return render_template("result.html",
          result=label,
          prob=f"{prob*100:.2f}%",
          upload_fn=fn,
          brand_name="SkinGuardian"
        )
    return render_template("index.html", brand_name="SkinGuardian")


@app.route("/interpret/<upload_fn>")
def interpret(upload_fn):
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_fn)
    pil      = Image.open(img_path)
    arr      = preprocess_onnx(pil)
    label, prob = run_inference(arr)

    occ_map  = compute_occlusion(arr)
    cam_map  = compute_gradcam(arr)
    lime_map = compute_lime(pil)

    return render_template("explain.html",
      result=label,
      prob=prob,
      occlusion=occ_map,
      gradcam=cam_map,
      lime=lime_map,
      brand_name="SkinGuardian"
    )


@app.route("/about")
def about():
    return render_template("about.html", brand_name="SkinGuardian")

@app.route("/contact")
def contact():
    return render_template("contact.html", brand_name="SkinGuardian")


def start_server():
    """Helper function if needed for PyInstaller."""
    app.run(debug=False, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    # Normal local development: python app.py
    app.run(debug=True)
