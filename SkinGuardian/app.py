import io, os, uuid
from flask import Flask, request, render_template
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
from skimage.segmentation import slic, find_boundaries

# Matplotlib “Agg” backend (no GUI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# (Optional) HuggingFace BEiT extractor
from transformers import BeitFeatureExtractor

# ─── Flask setup ───────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ─── Load ONNX + convert to PyTorch ─────────────────────────────────────────────
onnx_path = os.path.join(BASE_DIR, "beit_finetuned_model.onnx")
ort_sess  = ort.InferenceSession(onnx_path)
onnx_model = onnx.load(onnx_path)
pt_model   = convert(onnx_model).eval().to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
feat_ext = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224")

# ─── Preprocessing & Inference ─────────────────────────────────────────────────
def preprocess_onnx(pil: Image.Image) -> np.ndarray:
    img = pil.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2,0,1)[None]

def run_inference(arr: np.ndarray):
    inp_name = ort_sess.get_inputs()[0].name
    logits   = ort_sess.run(None, {inp_name: arr})[0]
    exps     = np.exp(logits)
    probs    = exps / exps.sum(axis=1, keepdims=True)
    p_cancer = float(probs[0,1])
    label    = "Positive" if p_cancer>0.5 else "Negative"
    return label, p_cancer

# ─── 1) Grad-CAM + grid overlay ────────────────────────────────────────────────
def gradcam_metrics(grayscale: np.ndarray, percentile: float = 95):
    thresh = np.percentile(grayscale, percentile)
    mask   = grayscale >= thresh
    frac   = mask.mean()
    return {
        "top_percentile": percentile,
        "area_fraction":  round(frac * 100, 2)
    }

def compute_gradcam(arr: np.ndarray):
    device = next(pt_model.parameters()).device

    # reshape for ViT
    def reshape_transform(tensor, height=14, width=14):
        x = tensor[:,1:,:].reshape(tensor.size(0), height, width, tensor.size(2))
        return x.permute(0,3,1,2)

    class Wrapper(torch.nn.Module):
        def __init__(self, gm): super().__init__(); self.gm = gm
        def forward(self, x):      return self.gm(x)

    wrapper = Wrapper(pt_model).to(device).eval()
    inp     = torch.from_numpy(arr).float().to(device)

    with torch.no_grad():
        logits = wrapper(inp)
    pred_idx = int(logits.argmax(dim=1).item())
    targets  = [ClassifierOutputTarget(pred_idx)]

    layers = [m for n,m in pt_model.named_modules() if "encoder" in n and "layer" in n]
    cam    = GradCAM(model=wrapper, target_layers=[layers[-1]], reshape_transform=reshape_transform)
    grayscale = cam(inp, targets=targets)[0]

    # — save standard heatmap overlay —
    orig    = arr[0].transpose(1,2,0)
    overlay = show_cam_on_image(orig, grayscale, use_rgb=True)
    gradcam_out = os.path.join("static","explain","gradcam.png")
    plt.imsave(gradcam_out, overlay)

    # — compute quantitative metric —
    metrics = gradcam_metrics(grayscale)

    # — new: outline top-percentile region on original —
    thr        = np.percentile(grayscale, metrics["top_percentile"])
    mask       = grayscale >= thr
    boundaries = find_boundaries(mask)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(orig)
    ax.imshow(boundaries, cmap="Reds", alpha=0.6)
    ax.axis("off")
    grid_out = os.path.join("static","explain","gradcam_grid.png")
    fig.savefig(grid_out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return "explain/gradcam.png", metrics, "explain/gradcam_grid.png"

# ─── 2) Occlusion + grid overlay ───────────────────────────────────────────────
def compute_occlusion(arr: np.ndarray, patch_size=32, stride=16):
    H, W     = arr.shape[2], arr.shape[3]
    baseline = arr.mean()

    _, orig_p = run_inference(arr)
    drops     = []
    heatmap   = np.zeros((H, W), dtype=np.float32)

    # occlude & record
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            tmp = arr.copy()
            tmp[0,:,y:y+patch_size,x:x+patch_size] = baseline
            _, p = run_inference(tmp)

            # old‐style heatmap
            y1, x1 = min(y+patch_size, H), min(x+patch_size, W)
            heatmap[y:y1, x:x1] = p

            drops.append((round(orig_p - p,4), y, x))

    # save standard occlusion heatmap
    heatmap     = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min()+1e-8)
    orig        = arr[0].transpose(1,2,0)
    fig, ax     = plt.subplots(figsize=(4,4))
    ax.imshow(orig, alpha=0.8)
    ax.imshow(heatmap, cmap="jet", alpha=0.4)
    ax.axis("off")
    occl_out    = os.path.join("static","explain","occlusion.png")
    fig.savefig(occl_out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # top-3 drops
    drops.sort(reverse=True, key=lambda t: t[0])
    top3 = drops[:3]

    # new: draw red boxes around those 3 patches
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(orig)
    for drop, y, x in top3:
        rect = patches.Rectangle((x, y), patch_size, patch_size,
                                 linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
    ax.axis("off")
    grid_out = os.path.join("static","explain","occlusion_grid.png")
    fig.savefig(grid_out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return "explain/occlusion.png", top3, "explain/occlusion_grid.png"

# ─── 3) LIME + grid overlay ───────────────────────────────────────────────────
def lime_metrics(exp, num_features=3):
    feats = exp.local_exp[exp.top_labels[0]]
    feats = sorted(feats, key=lambda f: abs(f[1]), reverse=True)[:num_features]
    return [{"id": int(sp), "weight": round(w,4)} for sp,w in feats]

def compute_lime(pil: Image.Image):
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

    mask    = exp.get_image_and_mask(exp.top_labels[0],
                                     positive_only=True,
                                     num_features=3,
                                     hide_rest=True)[1] > 0

    overlay = img_np.astype(np.float32)/255.0
    red     = np.zeros_like(overlay); red[...,0]=1.0
    alpha   = 0.6
    overlay[mask]   = overlay[mask]*(1-alpha) + red[mask]*alpha
    overlay[~mask] *= 0.4

    lime_out = os.path.join("static","explain","lime.png")
    Image.fromarray((overlay*255).astype(np.uint8)).save(lime_out)

    metrics = lime_metrics(exp)

    # new: superpixel grid + top-3 outlines
    segments   = slic(img_np, n_segments=100, compactness=10, sigma=1)
    boundaries = find_boundaries(segments)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(img_np)
    ax.imshow(boundaries, cmap="gray", alpha=0.6)
    for feat in metrics:
        mask_sp = (segments == feat["id"])
        ax.contour(mask_sp, levels=[0.5], colors="red", linewidths=1.2)
    ax.axis("off")
    grid_out = os.path.join("static","explain","lime_grid.png")
    fig.savefig(grid_out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return "explain/lime.png", metrics, "explain/lime_grid.png"



# ─── 4) Flask routes ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def index():
    if request.method=="POST" and (f := request.files.get("file")):
        pil = Image.open(io.BytesIO(f.read()))
        arr = preprocess_onnx(pil)
        label, prob = run_inference(arr)

        fn = f"{uuid.uuid4().hex}.png"
        pil.save(os.path.join(app.config["UPLOAD_FOLDER"], fn))
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
    if label != "Positive":
        prob = 1 - prob

    # unpack three explainers + grid paths
    gradcam_map, gradcam_metrics, gradcam_grid     = compute_gradcam(arr)
    occl_map, occlusion_top3, occlusion_grid       = compute_occlusion(arr)
    lime_map, lime_metrics, lime_grid              = compute_lime(pil)

    return render_template("explain.html",
        result=label,
        prob=prob,
        gradcam=gradcam_map,
        gradcam_grid=gradcam_grid,
        gradcam_metrics=gradcam_metrics,
        occlusion=occl_map,
        occlusion_grid=occlusion_grid,
        occlusion_top3=occlusion_top3,
        lime=lime_map,
        lime_grid=lime_grid,
        lime_metrics=lime_metrics,
        brand_name="SkinGuardian"
    )

@app.route("/about")
def about():
    return render_template("about.html", brand_name="SkinGuardian")

@app.route("/contact")
def contact():
    return render_template("contact.html", brand_name="SkinGuardian")

if __name__ == "__main__":
    app.run(debug=True)
