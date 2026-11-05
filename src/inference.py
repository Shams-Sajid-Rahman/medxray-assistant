"""
Inference for the chest X-ray assistant.

analyze(image) -> (label, confidence, pneumonia_prob, heatmap_overlay)
  label            "NORMAL" or "PNEUMONIA"
  confidence       probability of the predicted class (0-1)
  pneumonia_prob   probability of PNEUMONIA (0-1)
  heatmap_overlay  RGB numpy array (uint8) with the Grad-CAM heatmap on the X-ray
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import (
    BinaryClassifierOutputTarget,
    ClassifierOutputTarget,
)

# ---------------------------------------------------------------------------
# Settings. These MUST match what you used in Colab for validation/test.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # medxray-assistant/
MODEL_PATH = PROJECT_ROOT / "model" / "model.pt"

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet values (standard for DenseNet121)
STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # ImageFolder sorts folders alphabetically
THRESHOLD = 0.5                        # pneumonia_prob > THRESHOLD -> PNEUMONIA (same as notebook)

DEVICE = torch.device("cpu")           # laptops and free HF Spaces have no GPU

# Same as eval_tfm in the Colab notebook (medxray_training.ipynb)
_preprocess = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Loaded once, on the first call to analyze(), then reused.
_model = None
_cam = None
_num_outputs = None


def _clean_state_dict(sd):
    """Handle common ways a checkpoint can be wrapped."""
    for key in ("model_state_dict", "state_dict"):
        if key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    # "module." appears when a model was trained with DataParallel
    return {k.removeprefix("module."): v for k, v in sd.items()}


def _build_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    try:
        # Safe mode: works if you saved model.state_dict()
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except Exception:
        # You saved the whole model with torch.save(model, ...).
        # weights_only=False runs pickle code, so only use it with your OWN file.
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(ckpt, nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("model"), nn.Module):
        model = ckpt["model"]
    else:
        sd = _clean_state_dict(ckpt)
        if "classifier.weight" not in sd:
            head_keys = [k for k in sd if k.startswith("classifier")]
            raise KeyError(
                "Could not find 'classifier.weight' in the checkpoint. "
                f"Classifier keys found: {head_keys}. "
                "Send this message so the model-building code can be adjusted."
            )
        n_out = sd["classifier.weight"].shape[0]
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, n_out)
        model.load_state_dict(sd)

    model.to(DEVICE).eval()
    return model


def _load():
    global _model, _cam, _num_outputs
    if _model is None:
        _model = _build_model()
        with torch.no_grad():
            _num_outputs = _model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]
        # For torchvision DenseNet, the last layer of .features is the
        # standard Grad-CAM target (the final feature map before pooling).
        _cam = GradCAM(model=_model, target_layers=[_model.features[-1]])
    return _model, _cam, _num_outputs


def _to_pil(image):
    """Accept a file path, a PIL image, or a numpy array (what Gradio sends)."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")
    return image.convert("RGB")  # used for the overlay; _preprocess makes it grayscale x3


def analyze(image):
    model, cam, n_out = _load()
    pil = _to_pil(image)
    x = _preprocess(pil).unsqueeze(0).to(DEVICE)

    # 1) Prediction (no gradients needed)
    with torch.no_grad():
        logits = model(x)
    if n_out == 1:   # single output + sigmoid (BCE loss)
        pneumonia_prob = torch.sigmoid(logits)[0, 0].item()
    else:            # two outputs + softmax (CrossEntropy loss)
        pneumonia_prob = torch.softmax(logits, dim=1)[0, 1].item()

    pred_idx = int(pneumonia_prob > THRESHOLD)
    label = CLASS_NAMES[pred_idx]
    confidence = pneumonia_prob if pred_idx == 1 else 1.0 - pneumonia_prob

    # 2) Grad-CAM for the predicted class (needs gradients, so outside no_grad)
    target = (BinaryClassifierOutputTarget(pred_idx) if n_out == 1
              else ClassifierOutputTarget(pred_idx))
    grayscale_cam = cam(input_tensor=x, targets=[target])[0]

    # 3) Overlay the heatmap on the resized X-ray
    rgb = np.asarray(pil.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    return label, round(confidence, 4), round(pneumonia_prob, 4), overlay
