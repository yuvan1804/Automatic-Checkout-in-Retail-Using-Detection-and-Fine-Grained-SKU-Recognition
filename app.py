
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import io, os, random

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Automated Checkout in Retail Using Object Detection and Fine-Grained SKU Recognition-RPC",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');
:root{--bg:#0d0f13;--surface:#14171e;--border:#222636;--accent:#5c7cfa;
      --accent2:#38d9a9;--accent3:#f7b731;--text:#e8eaf0;--muted:#7a7f9a;--danger:#ff6b6b;}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;
  color:var(--text)!important;font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border);}
.shelf-header{display:flex;align-items:center;gap:14px;padding:28px 0 8px;
  border-bottom:1px solid var(--border);margin-bottom:28px;}
.shelf-logo{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;letter-spacing:-1px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.shelf-sub{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase;margin-top:2px;}
.pipe-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:28px;}
.pipe-node{font-family:'DM Mono',monospace;font-size:.68rem;padding:4px 10px;border-radius:4px;
  border:1px solid var(--border);color:var(--muted);background:var(--surface);letter-spacing:.5px;}
.pipe-arrow{color:var(--accent);font-size:.75rem;opacity:.6;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:20px 22px;margin-bottom:18px;}
.card-title{font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;
  text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:10px;}
.metric-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:22px;}
.metric-tile{flex:1;min-width:120px;background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:14px 16px;text-align:center;}
.metric-val{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:var(--accent);line-height:1;}
.metric-label{font-family:'DM Mono',monospace;font-size:.64rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:1px;margin-top:5px;}
.divider{height:1px;background:linear-gradient(90deg,var(--accent) 0%,transparent 70%);
  margin:24px 0;opacity:.35;}
.stButton>button{background:linear-gradient(135deg,var(--accent),#7c5cfc)!important;
  color:#fff!important;border:none!important;border-radius:6px!important;
  font-family:'Syne',sans-serif!important;font-weight:700!important;
  letter-spacing:.5px!important;padding:10px 26px!important;transition:opacity .2s;}
.stButton>button:hover{opacity:.85!important;}
label{color:var(--muted)!important;font-family:'DM Mono',monospace!important;font-size:.75rem!important;}
[data-testid="stFileUploader"]{border:1.5px dashed var(--border)!important;
  border-radius:10px!important;background:var(--surface)!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# BUG-3 FIX: VIT_CLASSES must EXACTLY match data.yaml (YOLO training order)
# Original had wrong order starting with 'milk'.
# ═══════════════════════════════════════════════════════════════
VIT_CLASSES = [
    'alcohol',          # 0   ← data.yaml class 0
    'candy',            # 1
    'canned_food',      # 2
    'chocolate',        # 3
    'dessert',          # 4
    'dried_food',       # 5
    'dried_fruit',      # 6
    'drink',            # 7
    'gum',              # 8
    'instant_drink',    # 9
    'instant_noodles',  # 10
    'milk',             # 11
    'personal_hygiene', # 12
    'puffed_food',      # 13
    'seasoner',         # 14
    'stationery',       # 15
    'tissue',           # 16
]

# Fallback hint-based candidate lists (used when gallery is absent)
_CAT_HINTS = {
    'milk':            ['Milk', 'Soy Milk', 'Yogurt', 'Dairy'],
    'alcohol':         ['Beer', 'Vodka', 'Whisky', 'Baijiu', 'Lager', 'Cocktail'],
    'instant_drink':   ['Tea Bags', 'Milk Tea', 'Oatmeal', 'Cereal', 'Soy Milk Powder',
                        'Congee Powder', 'Porridge'],
    'drink':           ['Coca-Cola', 'Pepsi', 'Sprite', 'Fanta', 'Mineral Water',
                        'Herbal Tea', 'Sparkling'],
    'dried_food':      ['Mushroom', 'Peanut', 'Pistachio', 'Cashew', 'Sunflower', 'Wood Ear'],
    'instant_noodles': ['Noodles', 'Instant Noodles', 'Yi Noodles'],
    'dried_fruit':     ['Mango', 'Peach', 'Lemon', 'Longan', 'Date', 'Goji', 'Wolfberry',
                        'Sweet Potato Strips'],
    'canned_food':     ['Canned', 'Congee Canned', 'Luncheon Meat', 'Coconut Cream', 'Fish Canned'],
    'stationery':      ['Glue', 'Marker', 'Notebook', 'Label', 'Tape', 'File Bag'],
    'tissue':          ['Tissue'],
    'gum':             ['Chewing Gum', 'Bubble Gum', 'Spearmint Gum'],
    'candy':           ['Candy', 'Lozenge', 'Throat', 'Caramel', 'Skittles', 'Starburst',
                        'Milk Tablet'],
    'personal_hygiene':['Toothpaste', 'Shampoo', 'Body Wash', 'Mouthwash', 'Laundry',
                        'Toothbrush', 'Styling Gel'],
    'seasoner':        ['Vinegar', 'Soy Sauce', 'Salt', 'Seasoning', 'Cooking Wine', 'Chili Powder'],
    'dessert':         ['Cake', 'Bread', 'Cookie', 'Biscuit', 'Wafer', 'Swiss Roll', 'Cupcake'],
    'chocolate':       ['Chocolate', 'Snickers', 'Dove', 'Kit Kat', 'Hersheys', 'MMs'],
    'puffed_food':     ['Crisps', 'Cracker', 'Puff Biscuits', 'Onion Rings', 'Prawn Sticks',
                        'Prawn Crackers', 'Cheetos', 'Oishi', 'Doritos', 'Panpan BBQ'],
}

# Fallback SKU data (used only when gallery unavailable)
_FALLBACK_SKUS = {
    'alcohol':         [('Pepsi Cola 600ml', 'Pepsi', '百事可乐600ml'),
                        ('Heineken Lager Beer 500ml Can', 'Heineken', '喜力啤酒500ml'),
                        ('Budweiser American Lager Beer 600ml', 'Budweiser', '百威啤酒600ml'),
                        ('Tsingtao Beer Classic Lager 330ml', 'Tsingtao', '青岛啤酒330ml')],
    'candy':           [('Skittles Original Fruit Flavor Chewy Candy 45g', 'Skittles', '彩虹糖原果味45g'),
                        ('Alps Caramel Milk Hard Candy 45g', 'Alps', '阿尔卑斯焦香牛奶味硬糖45g'),
                        ('Starburst Original Fruit Chews Assorted 25g', 'Starburst', '星爆缤纷原果味25g')],
    'canned_food':     [('Yinlu Barley Red Bean Congee Canned 280g', 'Yinlu', '银鹭薏仁红豆粥280g'),
                        ('Maling Luncheon Meat Canned Pork 340g', 'Maling', '梅林午餐肉340g'),
                        ('Dole Pineapple Chunks in Juice 567g', 'Dole', '都乐菠萝块567g')],
    'chocolate':       [('Dove Mango Yogurt Chocolate Bar 42g', 'Dove', '德芙芒果酸奶巧克力42g'),
                        ('Snickers Peanut Caramel Chocolate Bar 51g', 'Snickers', '士力架花生夹心巧克力51g'),
                        ('Hersheys Milk Chocolate Bar 40g', 'Hersheys', '好时牛奶巧克力40g')],
    'dessert':         [('Oreo Mini Chocolate Sandwich Cookies 55g', 'Oreo', 'mini奥利奥55g'),
                        ('Qinglian Pineapple Cream Sandwich Biscuits 63g', 'Qinglian', '庆联凤梨味夹心饼63g'),
                        ('Garden Strawberry Wafer Biscuits 50g', 'Garden', '嘉顿威化饼干草莓味50g')],
    'dried_food':      [('Huiyi Roasted Pistachio Nuts 140g', 'Huiyi', '惠宜开心果140g'),
                        ('Qiaqia Herbal Tea Flavor Sunflower Seeds 150g', 'Qiaqia', '洽洽凉茶瓜子150g'),
                        ('Huiyi Cashew Nuts 160g', 'Huiyi', '惠宜腰果160g')],
    'dried_fruit':     [('Huiyi Thailand Dried Mango Slices 80g', 'Huiyi', '惠宜泰国芒果干80g'),
                        ('Xinjiang Hetian Dried Red Dates 454g', 'Huiyi', '新疆和田滩枣454g'),
                        ('Huiyi Wolfberry Goji Berries 100g', 'Huiyi', '惠宜枸杞100g')],
    'drink':           [('Coca-Cola Classic 500ml', 'Coca-Cola', '可口可乐500ml'),
                        ('Pepsi Cola 600ml', 'Pepsi', '百事可乐600ml'),
                        ('Nongfu Spring Natural Mineral Water 550ml', 'Nongfu Spring', '农夫山泉矿泉水550ml'),
                        ('Sprite Lemon-Lime Sparkling Drink 500ml', 'Sprite', '雪碧500ml')],
    'gum':             [('Stride Spearmint Chewing Gum 21g', 'Stride', '炫迈薄荷味21g'),
                        ('Extra Spearmint Chewing Gum 5-Piece Pack 15g', 'Extra', '绿箭5片装15g')],
    'instant_drink':   [('Youlemei Taro Milk Tea Powder 80g', 'Youlemei', '优乐美香芋味80g'),
                        ('Lipton Lemon Flavor Tea Bags 180g', 'Lipton', '立顿柠檬风味茶180g'),
                        ('Quaker Mixed Berry Oatmeal 40g', 'Quaker', '桂格多种莓果麦片40g')],
    'instant_noodles': [('Master Kong Spicy Beef Instant Noodles 105g', 'Master Kong', '康师傅香辣牛肉面105g'),
                        ('Cup Noodles Seafood Flavor 84g', 'Cup Noodles', '合味道海鲜风味84g'),
                        ('Jinye Braised Beef Instant Noodles 114g', 'Jinye', '今野红烧牛肉面114g')],
    'milk':            [('Yili Pure Fresh Whole Milk 250ml', 'Yili', '伊利纯牛奶250ml'),
                        ('Wahaha AD Calcium Milk Drink 220g', 'Wahaha', '娃哈哈AD钙奶220g'),
                        ('Want Want Reconstituted Whole Milk 250ml', 'Want Want', '旺仔牛奶复原乳250ml')],
    'personal_hygiene':['Colgate Whitening Baking Soda Toothpaste 180g', 'Colgate', '高露洁亮白小苏打180g'],
    'puffed_food':     [('Oishi Prawn Crackers Original 40g', 'Oishi', '上好佳鲜虾片40g'),
                        ('Cheetos Japanese Steak Flavor 90g', 'Cheetos', '奇多日式牛排味90g'),
                        ('Doritos Magic Charcoal Grilled Flavor 65g', 'Doritos', '妙脆角魔力炭烧味65g')],
    'seasoner':        [('Hengshun Zhenjiang Aromatic Black Vinegar 340ml', 'Hengshun', '恒顺香醋340ml'),
                        ('Donggu Weijixian Superior Soy Sauce 150ml', 'Donggu', '东古味极鲜酱油150ml')],
    'stationery':      [('Guangbo Solid Glue Stick 15g', 'Guangbo', '广博固体胶15g'),
                        ('Morning Glory Snail Correction Tape', 'Morning Glory', '晨光蜗牛改正带')],
    'tissue':          [('Qingfeng Original Wood Pure Gold Edition Tissue 100x3', 'Qingfeng', '清风原木纯品金装100x3'),
                        ('Jierou Face Care Facial Tissue 150x3', 'Jierou', '洁柔face150x3')],
}

# ═══════════════════════════════════════════════════════════════
# MODEL PATHS — update to your paths
# ═══════════════════════════════════════════════════════════════
YOLO_PATH    = r"F:\sku_website\models\yolo_best.pt"
VIT_PATH     = r"F:\sku_website\models\vit_sku_best.pt"
GALLERY_PATH = r"F:\sku_website\models\sku_gallery.pt"

# ═══════════════════════════════════════════════════════════════
# BUG-1 FIX: correct model architecture (vit-ocr.ipynb Cell 5)
# ViTSKUEncoder on WinKawaks/vit-small-patch16-224, NOT torchvision.vit_b_16
# ═══════════════════════════════════════════════════════════════
class ViTSKUEncoder(nn.Module):
    """
    Exact reproduction of ViTSKUEncoder from vit-ocr.ipynb Cell 5.
      base      : WinKawaks/vit-small-patch16-224  (hidden_size=384)
      projector : Linear(768→512→256→embed_dim) with GELU+Dropout
      classifier: Linear(embed_dim → num_classes)
      forward   : CLS + mean(patch tokens) → projector → L2-norm → embed
    """
    def __init__(self, embed_dim=128, num_classes=200, use_patch_tokens=True):
        super().__init__()
        from transformers import ViTModel
        self.use_patch_tokens = use_patch_tokens
        self.vit    = ViTModel.from_pretrained('WinKawaks/vit-small-patch16-224')
        hidden      = self.vit.config.hidden_size          # 384 for vit-small
        proj_in     = hidden * 2 if use_patch_tokens else hidden   # 768 when True

        self.projector = nn.Sequential(
            nn.Linear(proj_in, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, pixel_values, return_embed=False):
        out       = self.vit(pixel_values=pixel_values)
        cls       = out.last_hidden_state[:, 0, :]
        pmean     = out.last_hidden_state[:, 1:, :].mean(dim=1)
        combined  = torch.cat([cls, pmean], dim=1) if self.use_patch_tokens else cls
        embed     = F.normalize(self.projector(combined), p=2, dim=1)
        if return_embed:
            return embed
        return embed, self.classifier(embed)


# ═══════════════════════════════════════════════════════════════
# LOADERS
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_yolo():
    try:
        from ultralytics import YOLO
        return YOLO(YOLO_PATH), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_vit_and_gallery():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ckpt = torch.load(VIT_PATH, map_location=dev)

        # Support three checkpoint formats
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            nc = ckpt.get("num_classes", 200)
            ed = ckpt.get("embed_dim", 128)
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            nc, ed = 200, 128
        else:
            # Raw state_dict — infer dims from classifier weight
            sd = ckpt
            cw = sd.get("classifier.weight", sd.get("module.classifier.weight"))
            nc, ed = (tuple(cw.shape) if cw is not None else (200, 128))

        # BUG-1 FIX: build ViTSKUEncoder, not torchvision vit_b_16
        model = ViTSKUEncoder(embed_dim=ed, num_classes=nc, use_patch_tokens=True)
        # BUG-7 FIX: strict=True catches weight mismatches immediately
        model.load_state_dict(sd, strict=True)
        model.to(dev).eval()

        # Load gallery
        gallery, gerr = None, None
        if os.path.exists(GALLERY_PATH):
            try:
                raw = torch.load(GALLERY_PATH, map_location=dev)
                lm  = raw.get("label_to_meta", None)
                if lm is None:
                    gerr = (
                        "label_to_meta is missing from sku_gallery.pt.\n"
                        "Re-run notebook Cell 8 with the NOTEBOOK PATCH shown in app.py docstring."
                    )
                gallery = {
                    "matrix":      raw["matrix"].to(dev),
                    "labels":      raw["labels"].to(dev),
                    "cat_indices": raw.get("sku_class_to_indices", {}),
                    "label_meta":  lm or {},          # BUG-5 FIX: use label_to_meta
                }
            except Exception as ge:
                gerr = str(ge)
        else:
            gerr = f"Gallery not found: {GALLERY_PATH}"

        return model, dev, ed, nc, gallery, gerr, None

    except Exception as e:
        return None, "cpu", 128, 200, None, None, str(e)


yolo_model, yolo_err = load_yolo()
vit_model, vit_device, vit_embed_dim, vit_num_classes, gallery, gallery_err, vit_err = \
    load_vit_and_gallery()

# ═══════════════════════════════════════════════════════════════
# BUG-2 FIX: normalisation must match notebook val_tf → [0.5]*3
# Original used ImageNet stats which differ from training.
# ═══════════════════════════════════════════════════════════════
_M, _S = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
_val_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(_M, _S)])

# TTA transforms matching notebook tta_transforms (Cell 3)
_tta_tfs = [
    T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(_M, _S)]),
    T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(p=1.0),
               T.ToTensor(), T.Normalize(_M, _S)]),
    T.Compose([T.Resize((int(224*1.1), int(224*1.1))), T.CenterCrop(224),
               T.ToTensor(), T.Normalize(_M, _S)]),
    T.Compose([T.Resize((224, 224)), T.RandomVerticalFlip(p=1.0),
               T.ToTensor(), T.Normalize(_M, _S)]),
]


# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_yolo(img_np, conf_thresh=0.25):
    """
    BUG-6 FIX: read box.cls → VIT_CLASSES[ci] supercategory per box.
    Original ignored box.cls entirely, so category was always guessed.
    """
    if yolo_model is None:
        return [], [], []
    boxes, confs, cats = [], [], []
    for r in yolo_model(img_np, conf=conf_thresh, verbose=False):
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append([x1, y1, x2, y2])
            confs.append(float(box.conf[0]))
            ci = int(box.cls[0].item())
            cats.append(VIT_CLASSES[ci] if ci < len(VIT_CLASSES) else "unknown")
    return boxes, confs, cats


def crop_regions(img_np, boxes):
    crops = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)
        c = img_np[y1:y2, x1:x2]
        crops.append(c if c.size > 0 else img_np[:10, :10])
    return crops


def embed_crop(crop_rgb, use_tta=True):
    """
    BUG-4 FIX: produce L2-normalised embedding (not softmax class scores).
    Matches notebook inference in Cell 11 exactly.
    """
    if vit_model is None:
        return None
    try:
        pil  = Image.fromarray(crop_rgb).convert("RGB")
        tfs  = _tta_tfs if use_tta else [_val_tf]
        embs = []
        with torch.no_grad():
            for tf in tfs:
                embs.append(vit_model(tf(pil).unsqueeze(0).to(vit_device), return_embed=True))
        return F.normalize(torch.stack(embs, 0).mean(0), p=2, dim=1)
    except Exception:
        return None


def gallery_retrieve(embed, yolo_cat):
    """
    BUG-4+5 FIX:
      - cosine similarity against pre-built gallery (not heuristic keyword lookup)
      - reads label_to_meta[lbl] keyed by integer gallery label (saved by patched Cell 8)
      - filters by YOLO supercategory when enough candidates exist
    """
    if gallery is None or embed is None:
        return None

    gm   = gallery["matrix"]
    gl   = gallery["labels"]
    lm   = gallery["label_meta"]
    idxs = gallery["cat_indices"].get(yolo_cat, [])

    # Use category-filtered sub-gallery when there are enough entries
    if len(idxs) >= 3:
        sims = torch.matmul(embed, gm[idxs].T).squeeze(0)
        bi   = sims.argmax().item()
        sim  = sims[bi].item()
        lbl  = int(gl[idxs[bi]].item())
    else:
        sims = torch.matmul(embed, gm.T).squeeze(0)
        bi   = sims.argmax().item()
        sim  = sims[bi].item()
        lbl  = int(gl[bi].item())

    # BUG-5 FIX: label_to_meta keyed by integer label (from patched Cell 8)
    meta = lm.get(lbl, lm.get(str(lbl), {}))
    return {
        "name":   meta.get("name",    f"SKU-{lbl}"),
        "brand":  meta.get("brand",   "Unknown"),
        "sku_cn": meta.get("sku_cn",  meta.get("name", "—")),
        "sim":    round(sim, 4),
    }


def fallback_lookup(cat):
    """
    BUG-8 FIX: random pick from category candidates, not always the first entry.
    Used only when gallery is unavailable.
    """
    cands = _FALLBACK_SKUS.get(cat, [])
    if cands:
        item = random.choice(cands)
        # Handle both tuple (name, brand, sku_cn) and bare string entries
        if isinstance(item, tuple):
            return item[0], item[1], item[2]
        return item, "Unknown", "—"
    return f"Unmatched ({cat})", "Unknown", "—"


def classify_and_retrieve(crop_rgb, yolo_cat, yolo_conf, use_tta=True):
    embed = embed_crop(crop_rgb, use_tta)
    ret   = gallery_retrieve(embed, yolo_cat) if embed is not None else None

    if ret:
        vit_conf           = ret["sim"]
        name, brand, sku_cn = ret["name"], ret["brand"], ret["sku_cn"]
    else:
        # No gallery — use fallback
        vit_conf = 0.0
        name, brand, sku_cn = fallback_lookup(yolo_cat)

    hybrid = round(0.6 * yolo_conf + 0.4 * vit_conf, 4)
    return {
        "Supercategory": yolo_cat,
        "Brand":         brand,
        "Product Name":  name,
        "SKU (CN)":      sku_cn,
        "Score":         hybrid,
        "YOLO Conf":     round(yolo_conf, 3),
        "ViT Conf":      round(vit_conf, 3),
    }


def count_products(results):
    c = Counter(r["Product Name"] for r in results)
    for r in results:
        r["Count"] = c[r["Product Name"]]
    return results


CAT_COLORS = {
    'milk':            (200, 230, 255),
    'alcohol':         (180, 105, 255),
    'instant_drink':   (100, 200, 255),
    'drink':           (92,  124, 250),
    'dried_food':      (56,  217, 169),
    'instant_noodles': (255, 165,  82),
    'dried_fruit':     (247, 183,  49),
    'canned_food':     (134, 142, 255),
    'stationery':      (160, 218, 169),
    'tissue':          (230, 230, 230),
    'gum':             (255, 180, 220),
    'candy':           (255, 100, 100),
    'personal_hygiene':(100, 230, 200),
    'seasoner':        (200, 150,  80),
    'dessert':         (255, 200, 120),
    'chocolate':       ( 80,  50,  20),
    'puffed_food':     (200, 255, 150),
}


def draw_detections(img_np, boxes, results):
    """BUG-8 FIX: label shows real brand from gallery, not hardcoded 'Unknown'."""
    out = img_np.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if i >= len(results):
            break
        cat   = results[i].get("Supercategory", "drink")
        color = CAT_COLORS.get(cat, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        brand = results[i].get("Brand", "?")
        score = results[i].get("Score", 0)
        lbl   = f"{brand} [{cat}] {score:.2f}"
        lx, ly = x1, max(y1 - 8, 12)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 4, ly + 2), color, -1)
        cv2.putText(out, lbl, (lx + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
    background:linear-gradient(135deg,#5c7cfa,#38d9a9);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;margin-bottom:4px;">Automated Checkout in Retail Using Object Detection and Fine-Grained SKU Recognition</div>
    <div style="font-family:'DM Mono',monospace;font-size:.65rem;color:#7a7f9a;
    letter-spacing:2px;text-transform:uppercase;margin-bottom:24px;">
    YOLO + ViT Embedding Retrieval<br><br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Model Status**")
    if yolo_model:
        st.success("YOLO loaded")
    else:
        st.error(f" YOLO: {yolo_err}")

    if vit_model:
        st.success(f" ViTSKUEncoder ({vit_device.upper()}) — {vit_num_classes} classes, dim={vit_embed_dim}")
    else:
        st.error(f" ViT: {vit_err}")

    if gallery and gallery.get("label_meta"):
        n_emb = gallery["matrix"].shape[0]
        st.success(f" Gallery: {n_emb} SKU embeddings + label_to_meta ✓")
    elif gallery:
        st.warning(" Gallery loaded BUT label_to_meta missing.\n"
                   "Re-run notebook Cell 8 with NOTEBOOK PATCH (see app.py docstring).")
    else:
        st.error(f" Gallery: {gallery_err}")
        st.info("Fallback: random candidate per category.\n"
                "For accurate SKU retrieval: regenerate sku_gallery.pt with NOTEBOOK PATCH.")

    st.markdown("---")
    st.markdown("**Detection Settings**")
    yolo_conf_thresh = st.slider("YOLO Confidence", 0.10, 0.90, 0.25, 0.05)
    sim_threshold    = st.slider("Score Display Threshold", 0.0, 1.0, 0.40, 0.05)
    max_dets         = st.slider("Max Detections", 3, 50, 20)
    use_tta          = st.checkbox("TTA — 4 augments (slower, more accurate)", True)
    st.markdown("---")
    st.markdown("**Display Settings**")
    show_unknown = st.checkbox("Include LOW confidence rows", True)
    show_sku_cn  = st.checkbox("Show Chinese SKU column", False)

    st.markdown("---")
    st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:.65rem;color:#7a7f9a;line-height:1.8;">
    PIPELINE<br>
    Image → YOLO (box+cls) → Crop<br>
    → ViTSKUEncoder embed + TTA<br>
    → Cosine Gallery Retrieval<br>
    → label_to_meta → Results<br><br>
    MODELS<br>
    · yolo_best.pt  (YOLO detection)<br>
    · vit_sku_best.pt  (ViTSKUEncoder)<br>
    · sku_gallery.pt  (embedding DB)
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
st.markdown("""<div class="shelf-header"><div>
  <div class="shelf-logo">Automated Checkout in Retail Using Object Detection and Fine-Grained SKU Recognition</div>
  <div class="shelf-sub">YOLO + ViT Embedding Retrieval</div>
</div></div>
<div class="pipe-row">
  <span class="pipe-node">Upload</span><span class="pipe-arrow">→</span>
  <span class="pipe-node">YOLO (box + cls)</span><span class="pipe-arrow">→</span>
  <span class="pipe-node">Crop (RGB)</span><span class="pipe-arrow">→</span>
  <span class="pipe-node">ViTSKUEncoder + TTA</span><span class="pipe-arrow">→</span>
  <span class="pipe-node">Cosine Gallery</span><span class="pipe-arrow">→</span>
  <span class="pipe-node">label_to_meta</span><span class="pipe-arrow">→</span>
  <span class="pipe-node">Results + Count</span>
</div>""", unsafe_allow_html=True)

uploaded = st.file_uploader("Drop a tray image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.markdown("""<div class="card" style="text-align:center;padding:50px 20px;">
      <div style="font-size:2.5rem;margin-bottom:12px;">🛒</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#7a7f9a;">
        Upload a retail tray image to begin</div>
      <div style="font-family:'DM Mono',monospace;font-size:.7rem;color:#4a4f65;margin-top:6px;">
        Supports JPG · PNG · multi-product scenes</div></div>""", unsafe_allow_html=True)
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_np  = np.array(pil_img)  # RGB uint8

c1, c2 = st.columns([3, 1], gap="large")
with c1:
    st.markdown('<div class="card-title">Uploaded Image</div>', unsafe_allow_html=True)
    st.image(pil_img, use_container_width=True)
with c2:
    st.markdown(f"""<div class="card"><div class="card-title">Info</div>
      <div style="font-family:'DM Mono',monospace;font-size:.78rem;line-height:2;color:#b0b5cc;">
        W: {pil_img.width}px<br>H: {pil_img.height}px<br>
        Mode: {pil_img.mode}<br>Size: {round(uploaded.size/1024,1)} KB</div></div>""",
        unsafe_allow_html=True)
    run_btn = st.button("⚡ Run Detection", use_container_width=True)

if run_btn:
    if yolo_model is None or vit_model is None:
        st.error(" One or more models failed to load. Check the sidebar for details.")
        st.stop()

    with st.spinner("Running pipeline…"):
        prog = st.progress(0, text="YOLO detection…")

        # Step 1: YOLO — boxes + confs + supercategory per box (BUG-6 FIX)
        boxes, yolo_confs, yolo_cats = run_yolo(img_np, yolo_conf_thresh)
        boxes, yolo_confs, yolo_cats = boxes[:max_dets], yolo_confs[:max_dets], yolo_cats[:max_dets]

        if not boxes:
            st.warning("No products detected. Try lowering the YOLO Confidence Threshold.")
            st.stop()
        prog.progress(30, text=f"{len(boxes)} boxes → ViT embedding…")

        # Step 2: Crop → embed → gallery retrieval
        crops   = crop_regions(img_np, boxes)
        results = []
        for i, (crop, cat) in enumerate(zip(crops, yolo_cats)):
            yc   = yolo_confs[i] if i < len(yolo_confs) else 0.5
            meta = classify_and_retrieve(crop, cat, yc, use_tta)
            meta["Confidence"] = ("HIGH" if meta["Score"] >= 0.80
                                  else ("MED" if meta["Score"] >= sim_threshold else "LOW"))
            results.append(meta)
            prog.progress(30 + int((i + 1) / len(crops) * 55),
                          text=f"Crop {i+1}/{len(crops)}…")

        results   = count_products(results)
        annotated = draw_detections(img_np, boxes, results)
        prog.progress(100, text="Done ✓")
        prog.empty()

    # Metrics
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    nt  = len(results)
    nhi = sum(1 for r in results if r["Confidence"] == "HIGH")
    nc  = len(set(r["Supercategory"] for r in results))
    avg = round(sum(r["Score"] for r in results) / max(nt, 1), 3)

    st.markdown(f"""<div class="metric-row">
      <div class="metric-tile"><div class="metric-val">{nt}</div>
        <div class="metric-label">Detections</div></div>
      <div class="metric-tile"><div class="metric-val" style="color:#38d9a9;">{nhi}</div>
        <div class="metric-label">High Confidence</div></div>
      <div class="metric-tile"><div class="metric-val" style="color:#f7b731;">{nc}</div>
        <div class="metric-label">Categories</div></div>
      <div class="metric-tile"><div class="metric-val" style="color:#5c7cfa;">{avg}</div>
        <div class="metric-label">Avg Score</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="card-title">Detection Output</div>', unsafe_allow_html=True)
    st.image(annotated, use_container_width=True,
             caption="Boxes coloured by YOLO supercategory — brand from gallery retrieval")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        acs = sorted(set(r["Supercategory"] for r in results))
        sc  = st.multiselect("Filter by category", acs, default=acs)
    with fc2:
        sconf = st.multiselect("Filter by confidence", ["HIGH", "MED", "LOW"],
                               default=["HIGH", "MED", "LOW"])

    df  = pd.DataFrame(results)
    if not show_sku_cn:
        df = df.drop(columns=["SKU (CN)"], errors="ignore")
    fdf = df[df["Supercategory"].isin(sc) & df["Confidence"].isin(sconf)]
    if not show_unknown:
        fdf = fdf[fdf["Confidence"] != "LOW"]

    st.markdown(f'<div class="card-title">Results · {len(fdf)} rows</div>',
                unsafe_allow_html=True)
    st.dataframe(fdf.style.background_gradient(subset=["Score"], cmap="Blues"),
                 use_container_width=True, hide_index=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Product Count Summary</div>', unsafe_allow_html=True)
    cnt = (fdf.groupby(["Supercategory", "Brand", "Product Name"])
           .agg(Count=("Count", "max"), Avg_Score=("Score", "mean"))
           .reset_index().sort_values("Count", ascending=False))
    cnt["Avg_Score"] = cnt["Avg_Score"].round(3)
    cc1, cc2 = st.columns([3, 2], gap="large")
    with cc1:
        st.dataframe(cnt, use_container_width=True, hide_index=True)
    with cc2:
        st.bar_chart(cnt.set_index("Product Name")["Count"])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇ Download CSV", fdf.to_csv(index=False).encode(),
                           "automated_checkout_results.csv", "text/csv", use_container_width=True)
    with d2:
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="PNG")
        st.download_button("⬇ Download Annotated Image", buf.getvalue(),
                           "automated_checkout_annotated.png", "image/png", use_container_width=True)