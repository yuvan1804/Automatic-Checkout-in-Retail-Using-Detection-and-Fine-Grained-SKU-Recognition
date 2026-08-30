
# ─────────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
import os, io, random
from collections import Counter, defaultdict

import streamlit as st

st.set_page_config(
    page_title="Automated Checkout in Retail Using Object Detection and Fine-Grained SKU Recognition-Indian Grocery",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# 1.  CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:      #0d0f13;
    --surface: #14171e;
    --border:  #222636;
    --accent:  #f97316;   /* warm saffron — India palette */
    --accent2: #22c55e;   /* fresh green  */
    --accent3: #facc15;   /* turmeric yellow */
    --danger:  #ef4444;
    --text:    #f1f5f9;
    --muted:   #7a7f9a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* ── Header ── */
.ig-header {
    padding: 28px 0 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.ig-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--accent3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}
.ig-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Pipeline badges ── */
.pipe-row {
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
    margin-bottom: 26px;
}
.pipe-node {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    padding: 4px 9px;
    border-radius: 4px;
    border: 1px solid var(--border);
    color: var(--muted);
    background: var(--surface);
    white-space: nowrap;
}
.pipe-arrow { color: var(--accent); font-size: 0.7rem; opacity: 0.55; }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.80rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    color: var(--muted);
    margin-bottom: 10px;
}

/* ── Metric tiles ── */
.metric-row  { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 22px; }
.metric-tile {
    flex: 1;
    min-width: 100px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 12px;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.60rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 5px;
}

/* ── Product card grid ── */
.product-grid { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
.product-card {
    flex: 1;
    min-width: 140px;
    max-width: 180px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 12px;
    text-align: center;
}
.product-emoji { font-size: 1.9rem; margin-bottom: 6px; }
.product-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.73rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 3px;
    line-height: 1.3;
}
.product-brand {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
}
.product-count {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    margin-top: 6px;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, var(--accent) 0%, transparent 70%);
    margin: 24px 0;
    opacity: 0.30;
}

/* ── Confidence pills ── */
.pill {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.70rem;
    font-weight: 500;
}
.pill-high { background: rgba(34,197,94,.15);  color: #22c55e; border: 1px solid rgba(34,197,94,.3); }
.pill-med  { background: rgba(250,204,21,.13);  color: #facc15; border: 1px solid rgba(250,204,21,.28); }
.pill-low  { background: rgba(239,68,68,.13);   color: #ef4444; border: 1px solid rgba(239,68,68,.28); }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #ea580c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 10px 26px !important;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.84 !important; }

/* ── Inputs ── */
label {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.73rem !important;
}
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
    background: var(--surface) !important;
}
[data-testid="stDataFrame"] table {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# 2.  HEAVY IMPORTS
# ─────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────
# 3.  DATASET — INDIAN GROCERY (10 products, exactly as in notebooks)
#
#     category_id  = YOLO class id (0-9, matches data.yaml names order)
#     label        = 0-indexed gallery label (same as category_id here,
#                    since cat_to_label = {i: i for i in range(10)})
#     sku_class    = supercategory used for gallery filtering
# ─────────────────────────────────────────────────────────────────

INDIAN_GROCERY_BRANDS = [
    {"category_id": 0, "name": "Bournvita",                             "brand": "Bournvita",                            "sku_class": "MaltDrink"},
    {"category_id": 1, "name": "Mysore Sandal Soap",                    "brand": "Mysore Sandal",                        "sku_class": "Soap"},
    {"category_id": 2, "name": "Nescafe Classic Coffee",                "brand": "Nescafe",                              "sku_class": "Coffee"},
    {"category_id": 3, "name": "Nivea Body Lotion",                     "brand": "Nivea",                                "sku_class": "Lotion"},
    {"category_id": 4, "name": "Nivea Soft Moisturising Cream",         "brand": "Nivea",                                "sku_class": "Moisturiser"},
    {"category_id": 5, "name": "Parachute Coconut Oil",                 "brand": "Parachute",                            "sku_class": "coconut Oil"},
    {"category_id": 6, "name": "Patanjali Dant Kanti",                  "brand": "Patanjali",                            "sku_class": "Toothpaste"},
    {"category_id": 7, "name": "Society Tea Powder",                    "brand": "Society",                              "sku_class": "TeaPowder"},
    {"category_id": 8, "name": "Tresemme Hairfall Defense Conditioner", "brand": "Tresemme",                             "sku_class": "conditioner"},
    {"category_id": 9, "name": "Tresemme Hairfall Defense Shampoo",     "brand": "Tresemme",                             "sku_class": "shampoo"},
]

# ── Lookup tables (mirrors vit notebook Cell 3) ─────────────────
sku_lookup   = {row["category_id"]: row for row in INDIAN_GROCERY_BRANDS}
NUM_CLASSES  = len(INDIAN_GROCERY_BRANDS)          # 10
cat_to_label = {row["category_id"]: i for i, row in enumerate(INDIAN_GROCERY_BRANDS)}
label_to_cat = {v: k for k, v in cat_to_label.items()}

sku_class_to_cat_ids: dict = defaultdict(list)
for row in INDIAN_GROCERY_BRANDS:
    sku_class_to_cat_ids[row["sku_class"]].append(row["category_id"])

# ── YOLO class names (data.yaml names list, index = class_id) ──
#    Must match exactly what the YOLO model was trained with.
YOLO_CLASS_NAMES = [
    "Bournvita",                                    # 0
    "Mysore Sandal Soap",                           # 1
    "Nescafe_Classic_Coffee",                       # 2
    "Nivea Body Lotion",                            # 3
    "Nivea_Soft_Moisturising_cream",                # 4
    "Parachute coconut Oil",                        # 5
    "Patanjali Dant Kanti",                         # 6
    "Society_TeaPowder_plain",                      # 7
    "Tresemme_Hairfall_Defense_conditioner",        # 8
    "Tresemme_Hairfall_Defense_shampoo",            # 9
]


# ─────────────────────────────────────────────────────────────────
# 4.  VISUAL IDENTITY — emoji + colour per product
# ─────────────────────────────────────────────────────────────────
PRODUCT_EMOJI = {
    "MaltDrink":   "",
    "Soap":        "",
    "Coffee":      "",
    "Lotion":      "",
    "Moisturiser": "",
    "coconut Oil": "",
    "Toothpaste":  "",
    "TeaPowder":   "",
    "conditioner": "",
    "shampoo":     "",
}

# BGR colours for OpenCV bounding boxes (one per category_id)
CAT_COLORS_BGR = {
    0: (30,  144, 255),   # Bournvita          — Dodger Blue
    1: (0,   200, 100),   # Mysore Sandal Soap — Emerald
    2: (0,   120, 200),   # Nescafe            — Coffee Blue
    3: (255, 180,  60),   # Nivea Body Lotion  — Amber
    4: (200, 100, 255),   # Nivea Soft Cream   — Violet
    5: (255, 255, 100),   # Parachute          — Yellow
    6: (50,  210, 180),   # Patanjali          — Teal
    7: (255, 140,   0),   # Society Tea        — Dark Orange
    8: (180, 100, 220),   # Tresemme Cond.     — Purple
    9: (100, 200, 255),   # Tresemme Shampoo   — Sky
}


# ─────────────────────────────────────────────────────────────────
# 5.  CONSTANTS  (from vit-indian-grocery Cell 0)
# ─────────────────────────────────────────────────────────────────
IMG_SIZE        = 224
EMBED_DIM       = 128
YOLO_CONF_DEF   = 0.30
CROP_MIN_PX     = 24
ASPECT_MIN      = 0.15
ASPECT_MAX      = 6.00
CROP_PAD        = 10
SIM_HIGH        = 0.40   # HIGH confidence threshold
SIM_MED         = 0.30   # MED  confidence threshold


# ─────────────────────────────────────────────────────────────────
# 6.  MODEL PATHS  ← EDIT THESE BEFORE RUNNING
# ─────────────────────────────────────────────────────────────────
YOLO_PATH    = r"F:\sku_website\models\ig_best.pt"    # from YOLO notebook
VIT_PATH     = r"F:\sku_website\models\vit_indian_grocery_best.pt"     # from ViT notebook
GALLERY_PATH = r"F:\sku_website\models\ig_sku_gallery.pt"              # from ViT notebook Step 6


# ─────────────────────────────────────────────────────────────────
# 7.  MODEL ARCHITECTURE  (exact vit-indian-grocery notebook Cell 5)
# ─────────────────────────────────────────────────────────────────
class ViTSKUEncoder(nn.Module):
    """
    Exact copy of ViTSKUEncoder from vit-indian-grocery notebook Cell 5.
    Base: WinKawaks/vit-small-patch16-224 (hidden_size = 384)
    Projector: Linear(768→512→256→embed_dim)  ·  GELU · Dropout
    Classifier: Linear(embed_dim → num_classes=10)
    """
    def __init__(self, embed_dim: int = 128, num_classes: int = 10,
                 use_patch_tokens: bool = True):
        super().__init__()
        from transformers import ViTModel
        self.use_patch_tokens = use_patch_tokens
        self.vit    = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")
        hidden      = self.vit.config.hidden_size          # 384
        proj_in     = hidden * 2 if use_patch_tokens else hidden  # 768

        self.projector = nn.Sequential(
            nn.Linear(proj_in, 512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, 256),    nn.GELU(), nn.Dropout(0.10),
            nn.Linear(256, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, pixel_values, return_embed: bool = False,
                return_attentions: bool = False):
        out       = self.vit(pixel_values=pixel_values,
                             output_attentions=return_attentions)
        cls       = out.last_hidden_state[:, 0, :]
        pmean     = out.last_hidden_state[:, 1:, :].mean(dim=1)
        combined  = torch.cat([cls, pmean], dim=1) if self.use_patch_tokens else cls
        embed     = F.normalize(self.projector(combined), p=2, dim=1)
        if return_embed:
            return embed
        return embed, self.classifier(embed)


# ─────────────────────────────────────────────────────────────────
# 8.  TRANSFORMS  (exact vit-indian-grocery notebook Cell 2)
# ─────────────────────────────────────────────────────────────────
_M, _S = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

_val_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(_M, _S),
])

_tta_transforms = [
    T.Compose([T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)),
               T.ToTensor(), T.Normalize(_M, _S)]),
    T.Compose([T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)),
               T.RandomHorizontalFlip(p=1.0),
               T.ToTensor(), T.Normalize(_M, _S)]),
    T.Compose([T.ToPILImage(),
               T.Resize((int(IMG_SIZE * 1.1), int(IMG_SIZE * 1.1))),
               T.CenterCrop(IMG_SIZE),
               T.ToTensor(), T.Normalize(_M, _S)]),
    T.Compose([T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)),
               T.RandomVerticalFlip(p=1.0),
               T.ToTensor(), T.Normalize(_M, _S)]),
]


# ─────────────────────────────────────────────────────────────────
# 9.  MODEL LOADERS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo():
    try:
        from ultralytics import YOLO
        return YOLO(YOLO_PATH), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_vit():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sd = torch.load(VIT_PATH, map_location=dev)
        # Support raw state_dict or wrapped checkpoint
        if isinstance(sd, dict) and "model_state_dict" in sd:
            state = sd["model_state_dict"]
            nc    = sd.get("num_classes", NUM_CLASSES)
            ed    = sd.get("embed_dim", EMBED_DIM)
        elif isinstance(sd, dict) and "state_dict" in sd:
            state, nc, ed = sd["state_dict"], NUM_CLASSES, EMBED_DIM
        else:
            state, nc, ed = sd, NUM_CLASSES, EMBED_DIM

        model = ViTSKUEncoder(embed_dim=ed, num_classes=nc, use_patch_tokens=True)
        model.load_state_dict(state, strict=True)
        model.to(dev).eval()
        return model, dev, ed, nc, None
    except Exception as e:
        return None, "cpu", EMBED_DIM, NUM_CLASSES, str(e)


@st.cache_resource
def load_gallery():
    if not os.path.exists(GALLERY_PATH):
        return None, f"Gallery file not found: {GALLERY_PATH}"
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        g   = torch.load(GALLERY_PATH, map_location=dev)
        return {
            "matrix":               g["matrix"].to(dev),
            "labels":               g["labels"].to(dev),
            "sku_class_to_indices": g.get("sku_class_to_indices", {}),
        }, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────
# 10. INITIALISE RESOURCES
# ─────────────────────────────────────────────────────────────────
yolo_model,  yolo_err                        = load_yolo()
vit_model,   vit_dev, vit_ed, vit_nc, vit_err = load_vit()
gallery,     gallery_err                     = load_gallery()


# ─────────────────────────────────────────────────────────────────
# 11. IMAGE PRE-PROCESSING — CLAHE
# ─────────────────────────────────────────────────────────────────
def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────
# 12. YOLO DETECTION
# ─────────────────────────────────────────────────────────────────
def run_yolo(img_bgr: np.ndarray, conf_thresh: float, max_dets: int):
    if yolo_model is None:
        return [], [], []
    boxes, confs, cat_ids = [], [], []
    for result in yolo_model(img_bgr, conf=conf_thresh, imgsz=640, verbose=False):
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            w, h = x2 - x1, y2 - y1
            if w < CROP_MIN_PX or h < CROP_MIN_PX:
                continue
            if not (ASPECT_MIN <= w / max(h, 1) <= ASPECT_MAX):
                continue
            ci = int(box.cls[0].item())
            if ci >= NUM_CLASSES:
                continue
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            confs.append(float(box.conf[0]))
            cat_ids.append(ci)
            if len(boxes) >= max_dets:
                break
        if len(boxes) >= max_dets:
            break
    return boxes, confs, cat_ids


# ─────────────────────────────────────────────────────────────────
# 13. CROP EXTRACTION  (pad = 10 px, from ViT notebook Cell 2)
# ─────────────────────────────────────────────────────────────────
def extract_crop_rgb(img_bgr: np.ndarray, box: list) -> np.ndarray:
    x1, y1, x2, y2 = box
    IH, IW = img_bgr.shape[:2]
    crop = img_bgr[
        max(0, y1 - CROP_PAD): min(IH, y2 + CROP_PAD),
        max(0, x1 - CROP_PAD): min(IW, x2 + CROP_PAD),
    ]
    if crop.size == 0:
        crop = img_bgr[:32, :32]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────
# 14. VIT EMBEDDING  (4-view TTA — mirrors ViT notebook Cell 10)
# ─────────────────────────────────────────────────────────────────
def embed_crop(crop_rgb: np.ndarray, use_tta: bool = True):
    if vit_model is None:
        return None
    try:
        tfs  = _tta_transforms if use_tta else [_val_tf]
        embs = []
        with torch.no_grad():
            for tf in tfs:
                t = tf(crop_rgb).unsqueeze(0).to(vit_dev)
                embs.append(vit_model(t, return_embed=True))
        return F.normalize(torch.stack(embs).mean(0), p=2, dim=1)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# 15. GALLERY RETRIEVAL  (cosine similarity — ViT notebook Cell 10)
#
#     gallery['labels'] are 0-indexed (0-9).
#     label_to_cat[label]  →  category_id (0-9, same here).
#     sku_lookup[cat_id]   →  full product row.
# ─────────────────────────────────────────────────────────────────
def gallery_retrieve(embed, yolo_cat_id: int, top_k: int = 3):
    """Cosine similarity retrieval with sku_class pre-filtering."""
    if gallery is None or embed is None:
        return None
    gm, gl = gallery["matrix"], gallery["labels"]

    # Map YOLO cat_id → sku_class → gallery indices
    sku_class = sku_lookup[yolo_cat_id]["sku_class"]
    idxs      = gallery["sku_class_to_indices"].get(sku_class, [])

    if len(idxs) >= top_k:
        sims = torch.matmul(embed, gm[idxs].T).squeeze(0)
        bi   = sims.argmax().item()
        return {"label": int(gl[idxs[bi]].item()), "sim": round(sims[bi].item(), 4)}
    else:
        # Full gallery fallback
        sims = torch.matmul(embed, gm.T).squeeze(0)
        bi   = sims.argmax().item()
        return {"label": int(gl[bi].item()), "sim": round(sims[bi].item(), 4)}


def resolve_meta(gallery_label: int) -> dict:
    """gallery_label (0-9) → label_to_cat → cat_id → sku_lookup row."""
    cat_id = label_to_cat.get(gallery_label, gallery_label)
    m = sku_lookup.get(cat_id)
    if m:
        return m
    return {"name": f"Unknown (label {gallery_label})", "brand": "Unknown",
            "sku_class": "unknown", "category_id": gallery_label}


# ─────────────────────────────────────────────────────────────────
# 16. FALLBACK  (when gallery not loaded — use YOLO cat_id directly)
# ─────────────────────────────────────────────────────────────────
def yolo_fallback_meta(cat_id: int) -> dict:
    """Direct lookup: YOLO class_id → INDIAN_GROCERY_BRANDS row."""
    return sku_lookup.get(cat_id, {
        "name": YOLO_CLASS_NAMES[cat_id] if cat_id < len(YOLO_CLASS_NAMES) else "Unknown",
        "brand": "Unknown",
        "sku_class": "unknown",
        "category_id": cat_id,
    })


# ─────────────────────────────────────────────────────────────────
# 17. FULL PER-CROP PIPELINE
# ─────────────────────────────────────────────────────────────────
def identify_crop(crop_rgb: np.ndarray, yolo_cat_id: int, yolo_conf: float,
                  use_tta: bool = True) -> dict:
    embed = embed_crop(crop_rgb, use_tta)
    ret   = gallery_retrieve(embed, yolo_cat_id) if embed is not None else None

    if ret is not None:
        vit_sim = ret["sim"]
        meta    = resolve_meta(ret["label"])
    else:
        # No gallery → trust YOLO class_id directly
        vit_sim = 0.0
        meta    = yolo_fallback_meta(yolo_cat_id)

    has_gallery = gallery is not None
    score = round(0.6 * yolo_conf + 0.4 * vit_sim, 4) if has_gallery else round(yolo_conf, 4)
    tier  = "HIGH" if vit_sim >= SIM_HIGH else ("MED" if vit_sim >= SIM_MED else "LOW")

    return {
        "Category ID":   meta.get("category_id", yolo_cat_id),
        "SKU Class":     meta.get("sku_class", "unknown"),
        "Brand":         meta.get("brand", "Unknown"),
        "Product Name":  meta.get("name", "Unknown"),
        "Score":         score,
        "YOLO Conf":     round(yolo_conf, 3),
        "ViT Sim":       round(vit_sim, 3),
        "Confidence":    tier,
    }


def add_counts(results: list) -> list:
    c = Counter(r["Product Name"] for r in results)
    for r in results:
        r["Count"] = c[r["Product Name"]]
    return results


# ─────────────────────────────────────────────────────────────────
# 18. ANNOTATION DRAWING
# ─────────────────────────────────────────────────────────────────
def draw_detections(img_rgb: np.ndarray, boxes: list, results: list,
                    cat_ids: list) -> np.ndarray:
    out = img_rgb.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if i >= len(results):
            break
        r     = results[i]
        cid   = cat_ids[i] if i < len(cat_ids) else 0
        color = CAT_COLORS_BGR.get(cid, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        lbl = f"{r['Brand']} | {r['SKU Class']} | {r['Score']:.2f} [{r['Confidence']}]"
        lx, ly = x1, max(y1 - 8, 14)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.rectangle(out, (lx, ly - th - 3), (lx + tw + 4, ly + 2), color, -1)
        cv2.putText(out, lbl, (lx + 2, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (10, 10, 10), 1, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════
# ────────────────────────  U I  ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
    background:linear-gradient(135deg,#f97316,#facc15);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:2px;"> Indian Grocery</div>
    <div style="font-family:'DM Mono',monospace;font-size:.60rem;color:#7a7f9a;
    letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;">
    YOLO (CBAM+DyHead) + ViT-small · 10 Products</div>
    """, unsafe_allow_html=True)

    # ── Model Status ────────────────────────────────────────────
    st.markdown("**Model Status**")

    if yolo_model:
        st.success(" YOLO (CBAM+DyHead) loaded")
    else:
        st.error(f" YOLO: {yolo_err}")

    if vit_model:
        st.success(f" ViTSKUEncoder ({vit_dev.upper()}) — {vit_nc} classes · dim={vit_ed}")
    else:
        st.error(f" ViT: {vit_err}")

    if gallery is not None:
        n_emb = gallery["matrix"].shape[0]
        st.success(f" Gallery: {n_emb} embeddings ({NUM_CLASSES} brands)")
    else:
        st.warning(f" Gallery: {gallery_err}")
        st.info(" Fallback: YOLO class_id used directly for product lookup.")

    st.divider()

    # ── Dataset reference ───────────────────────────────────────
    st.markdown("**Dataset (10 Products)**")
    for row in INDIAN_GROCERY_BRANDS:
        emoji = PRODUCT_EMOJI.get(row["sku_class"], " ")
        st.markdown(
            f"<div style='font-family:DM Mono,monospace;font-size:.62rem;"
            f"color:#b0b5cc;margin-bottom:3px;'>"
            f"{emoji} <b>{row['category_id']}</b> · {row['name']}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Detection settings ──────────────────────────────────────
    st.markdown("**Detection Settings**")
    yolo_conf_thresh = st.slider("YOLO Confidence", 0.10, 0.90, YOLO_CONF_DEF, 0.05,
                                 help="YOLO_CONF_MIN = 0.30 (from notebook)")
    max_dets  = st.slider("Max Detections", 1, 30, 15)
    use_tta   = st.checkbox("4-View TTA (slower, more accurate)", value=True,
                            help="Base · HFlip · 1.1×CenterCrop · VFlip")
    use_clahe = st.checkbox("CLAHE Pre-processing", value=True)

    st.divider()

    # ── Display settings ────────────────────────────────────────
    st.markdown("**Display Settings**")
    score_thresh = st.slider("Min Score", 0.0, 1.0, 0.0, 0.05)
    show_low     = st.checkbox("Show LOW confidence rows", value=True)

    st.divider()

    # ── Pipeline legend ─────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:.60rem;
    color:#7a7f9a;line-height:2.0;">
    PIPELINE<br>
    Upload → CLAHE<br>
    → YOLO CBAM+DyHead (box+cls)<br>
    → Crop + 10px pad<br>
    → ViTSKUEncoder 4-view TTA<br>
    → Cosine sku_class filter<br>
    → label_to_cat → sku_lookup<br>
    → Hybrid Score = 0.6×YOLO + 0.4×ViT<br>
    <br>
    CONFIDENCE<br>
    HIGH  sim ≥ 0.40<br>
    MED   sim ≥ 0.30<br>
    LOW   sim &lt; 0.30
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ig-header">
  <div class="ig-logo">Automated Checkout in Retail Using Object Detection and Fine-Grained SKU Recognition · Indian Grocery</div>
  <div class="ig-sub">
      YOLOv8 (CBAM + DyHead) · ViT-small-p16 SKU Embedding · 10 Indian Products
  </div>
</div>

<div class="pipe-row">
  <span class="pipe-node">Upload</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">CLAHE</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">YOLO CBAM+DyHead</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">Crop + 10px</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">ViT TTA embed</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">Cosine gallery</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">label_to_cat</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">sku_lookup</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">Results</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PRODUCT CATALOGUE (always visible below header)
# ─────────────────────────────────────────────────────────────────
with st.expander("View Product Catalogue (10 items)", expanded=False):
    cols = st.columns(5)
    for i, row in enumerate(INDIAN_GROCERY_BRANDS):
        emoji = PRODUCT_EMOJI.get(row["sku_class"], " ")
        cid   = row["category_id"]
        r, g, b = reversed(CAT_COLORS_BGR.get(cid, (180, 180, 180)))
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        with cols[i % 5]:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:12px 8px;
            border-left:3px solid {hex_color};">
              <div style="font-size:1.6rem;">{emoji}</div>
              <div style="font-family:'Syne',sans-serif;font-size:.72rem;
              font-weight:700;color:{hex_color};margin:4px 0;line-height:1.3;">
                {row['name']}</div>
              <div style="font-family:'DM Mono',monospace;font-size:.60rem;
              color:#7a7f9a;">
                ID {cid} · {row['sku_class']}<br>{row['brand']}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a grocery tray image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload any tray image containing Indian grocery products",
)

if not uploaded:
    st.markdown("""
    <div class="card" style="text-align:center;padding:52px 20px;">
      <div style="font-size:3rem;margin-bottom:14px;"></div>
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
      font-weight:700;color:#7a7f9a;">
        Drop a grocery tray image to begin recognition
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:.68rem;
      color:#4a4f65;margin-top:8px;">
        Supports JPG · PNG · single or multi-product scenes
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────
# LOAD + PREVIEW
# ─────────────────────────────────────────────────────────────────
pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

col_img, col_ctrl = st.columns([3, 1], gap="large")

with col_img:
    st.markdown('<div class="card-title">Uploaded Image</div>', unsafe_allow_html=True)
    st.image(pil_img, use_container_width=True)

with col_ctrl:
    size_kb = round(uploaded.size / 1024, 1)
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Image Info</div>
      <div style="font-family:'DM Mono',monospace;font-size:.76rem;
      line-height:2.2;color:#b0b5cc;">
        <b>Width  :</b> {pil_img.width} px<br>
        <b>Height :</b> {pil_img.height} px<br>
        <b>Mode   :</b> {pil_img.mode}<br>
        <b>Size   :</b> {size_kb} KB
      </div>
    </div>
    """, unsafe_allow_html=True)
    run_btn = st.button(" Run Detection", use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────
if run_btn:
    if yolo_model is None or vit_model is None:
        st.error(
            " One or more models not loaded — update the model paths "
            "at the top of this file and restart the app."
        )
        st.stop()

    with st.spinner("Running Automated Checkout in Retail Using Object Detection and Fine-Grained SKU Recognition pipeline…"):
        prog = st.progress(0, text="Step 1/4 · CLAHE pre-processing…")

        # Step 1: CLAHE
        proc_bgr = preprocess_image(img_bgr) if use_clahe else img_bgr
        prog.progress(10, text="Step 2/4 · YOLO (CBAM+DyHead) detection…")

        # Step 2: YOLO
        boxes, yolo_confs, yolo_cat_ids = run_yolo(proc_bgr, yolo_conf_thresh, max_dets)

        if not boxes:
            st.warning(
                " No products detected. Try lowering the "
                "**YOLO Confidence Threshold** in the sidebar."
            )
            st.stop()

        prog.progress(28, text=f"Step 3/4 · {len(boxes)} crops → ViT embedding + TTA…")

        # Step 3: Per-crop identify
        results = []
        for i, (box, cid, yc) in enumerate(zip(boxes, yolo_cat_ids, yolo_confs)):
            crop_rgb = extract_crop_rgb(proc_bgr, box)
            res      = identify_crop(crop_rgb, cid, yc, use_tta=use_tta)
            results.append(res)
            prog.progress(28 + int((i + 1) / len(boxes) * 62),
                          text=f"Step 3/4 · Crop {i+1}/{len(boxes)}…")

        # Step 4: Annotate
        prog.progress(95, text="Step 4/4 · Drawing annotations…")
        results   = add_counts(results)
        annotated = draw_detections(img_rgb, boxes, results, yolo_cat_ids)
        prog.progress(100, text="Done ✓")
        prog.empty()

    # ══════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────
    n_total  = len(results)
    n_high   = sum(1 for r in results if r["Confidence"] == "HIGH")
    n_med    = sum(1 for r in results if r["Confidence"] == "MED")
    n_low    = sum(1 for r in results if r["Confidence"] == "LOW")
    n_prods  = len(set(r["Product Name"] for r in results))
    avg_sc   = round(sum(r["Score"] for r in results) / max(n_total, 1), 3)
    avg_vit  = round(sum(r["ViT Sim"] for r in results) / max(n_total, 1), 3)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-tile">
        <div class="metric-val">{n_total}</div>
        <div class="metric-label">Detections</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val" style="color:#22c55e;">{n_high}</div>
        <div class="metric-label">High Conf</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val" style="color:#facc15;">{n_med}</div>
        <div class="metric-label">Med Conf</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val" style="color:#ef4444;">{n_low}</div>
        <div class="metric-label">Low Conf</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val" style="color:#f97316;">{n_prods}</div>
        <div class="metric-label">Products Found</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val" style="color:#fb923c;">{avg_sc}</div>
        <div class="metric-label">Avg Score</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val" style="color:#a78bfa;">{avg_vit}</div>
        <div class="metric-label">Avg ViT Sim</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Annotated image ───────────────────────────────────────
    st.markdown('<div class="card-title">Detection Output</div>', unsafe_allow_html=True)
    st.image(annotated, use_container_width=True,
             caption="Boxes coloured by product · brand resolved via ViT gallery retrieval")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Detected product tiles ────────────────────────────────
    st.markdown('<div class="card-title">Detected Products</div>', unsafe_allow_html=True)

    # Group by product name
    product_summary: dict = {}
    for r in results:
        name = r["Product Name"]
        if name not in product_summary:
            product_summary[name] = {
                "count":     0,
                "sku_class": r["SKU Class"],
                "brand":     r["Brand"],
                "cat_id":    r["Category ID"],
                "scores":    [],
            }
        product_summary[name]["count"] += 1
        product_summary[name]["scores"].append(r["Score"])

    # Render tiles
    tile_cols = st.columns(min(len(product_summary), 4))
    for j, (pname, pinfo) in enumerate(
        sorted(product_summary.items(), key=lambda x: -x[1]["count"])
    ):
        emoji   = PRODUCT_EMOJI.get(pinfo["sku_class"], " ")
        cid     = pinfo["cat_id"]
        r_, g_, b_ = reversed(CAT_COLORS_BGR.get(cid, (180, 180, 180)))
        hex_c   = f"#{r_:02x}{g_:02x}{b_:02x}"
        avg_s   = round(sum(pinfo["scores"]) / len(pinfo["scores"]), 3)

        with tile_cols[j % len(tile_cols)]:
            st.markdown(f"""
            <div class="card" style="text-align:center;border-top:3px solid {hex_c};">
              <div class="product-emoji">{emoji}</div>
              <div class="product-name">{pname}</div>
              <div class="product-brand">{pinfo['brand']} · {pinfo['sku_class']}</div>
              <div class="product-count" style="color:{hex_c};">{pinfo['count']}×</div>
              <div style="font-family:'DM Mono',monospace;font-size:.60rem;
              color:#7a7f9a;margin-top:4px;">avg score {avg_s}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────
    st.markdown('<div class="card-title">Filter & Analyse</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([2, 2, 1])

    with fc1:
        all_prods = sorted(set(r["Product Name"] for r in results))
        sel_prods = st.multiselect("Product", all_prods, default=all_prods)
    with fc2:
        tiers_av  = [t for t in ["HIGH", "MED", "LOW"]
                     if any(r["Confidence"] == t for r in results)]
        def_tiers = tiers_av if show_low else [t for t in tiers_av if t != "LOW"]
        sel_tiers = st.multiselect("Confidence", tiers_av, default=def_tiers)
    with fc3:
        all_skuclass = sorted(set(r["SKU Class"] for r in results))
        sel_class    = st.multiselect("SKU Class", all_skuclass, default=all_skuclass)

    # ── Results table ─────────────────────────────────────────
    df  = pd.DataFrame(results)
    fdf = df[
        df["Product Name"].isin(sel_prods) &
        df["Confidence"].isin(sel_tiers) &
        df["SKU Class"].isin(sel_class) &
        (df["Score"] >= score_thresh)
    ].reset_index(drop=True)

    display_cols = ["Category ID", "SKU Class", "Brand", "Product Name",
                    "Score", "YOLO Conf", "ViT Sim", "Confidence", "Count"]

    st.markdown(
        f'<div class="card-title">Detection Results · {len(fdf)} row{"s" if len(fdf) != 1 else ""}</div>',
        unsafe_allow_html=True,
    )

    if fdf.empty:
        st.info("No results match the current filters.")
    else:
        def _conf_style(val):
            styles = {"HIGH": "background:#0d2e1a;color:#22c55e;font-weight:600",
                      "MED":  "background:#2c2a10;color:#facc15;font-weight:600",
                      "LOW":  "background:#2e1212;color:#ef4444;font-weight:600"}
            return styles.get(val, "")

        styled = (
            fdf[display_cols]
            .style
            .background_gradient(subset=["Score"],    cmap="Oranges")
            .background_gradient(subset=["YOLO Conf"], cmap="Greens")
            .background_gradient(subset=["ViT Sim"],   cmap="Purples")
            .applymap(_conf_style, subset=["Confidence"])
            .format({"Score": "{:.3f}", "YOLO Conf": "{:.3f}", "ViT Sim": "{:.3f}"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Count Summary ─────────────────────────────────────────
    st.markdown('<div class="card-title">Product Count Summary</div>', unsafe_allow_html=True)

    if not fdf.empty:
        cnt = (
            fdf.groupby(["SKU Class", "Brand", "Product Name"])
            .agg(Count=("Count", "max"),
                 Avg_Score=("Score", "mean"),
                 Avg_ViT_Sim=("ViT Sim", "mean"))
            .reset_index()
            .sort_values("Count", ascending=False)
        )
        cnt["Avg_Score"]   = cnt["Avg_Score"].round(3)
        cnt["Avg_ViT_Sim"] = cnt["Avg_ViT_Sim"].round(3)

        cc1, cc2 = st.columns([3, 2], gap="large")
        with cc1:
            st.dataframe(cnt, use_container_width=True, hide_index=True)
        with cc2:
            if not cnt.empty:
                st.bar_chart(
                    cnt.set_index("Product Name")[["Count"]],
                    use_container_width=True,
                )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Confidence breakdown ──────────────────────────────────
    if not fdf.empty:
        st.markdown('<div class="card-title">Confidence Breakdown per Product</div>',
                    unsafe_allow_html=True)
        breakdown = (
            fdf.groupby(["Product Name", "Confidence"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        st.dataframe(breakdown, use_container_width=True, hide_index=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Exports ───────────────────────────────────────────────
    st.markdown('<div class="card-title">Export Results</div>', unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.download_button(
            "⬇ Download Results (CSV)",
            data=fdf.to_csv(index=False).encode("utf-8"),
            file_name="indian_grocery_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="PNG")
        st.download_button(
            "⬇ Download Annotated Image",
            data=buf.getvalue(),
            file_name="indian_grocery_annotated.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl3:
        if not fdf.empty and "cnt" in dir():
            st.download_button(
                "⬇ Download Count Summary",
                data=cnt.to_csv(index=False).encode("utf-8"),
                file_name="indian_grocery_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;font-family:'DM Mono',monospace;font-size:.60rem;
    color:#4a4f65;margin-top:36px;padding-bottom:20px;">
    Automated Checkout in Retail Using Object Detection and Fine-Grained SKU Recognition · Indian Grocery · Dataset: agentsk47/indian-grocery-object-detection-mfsnx · 10 products ·
    YOLO CBAM+DyHead → ViTSKUEncoder TTA → Cosine Gallery → label_to_cat → sku_lookup ·
    Score = 0.6×YOLO + 0.4×ViT
    </div>
    """, unsafe_allow_html=True)