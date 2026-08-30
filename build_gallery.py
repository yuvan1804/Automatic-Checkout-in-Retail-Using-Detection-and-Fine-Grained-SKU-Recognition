# """
# build_gallery.py
# Run ONCE after training to precompute the SKU embedding gallery.
# Saves:
#   data/sku_gallery_embeddings.npy  — shape (N, embed_dim) float32
#   data/sku_gallery_ids.npy         — shape (N,)            int32
#   data/sku_lookup.json             — {sku_id: {brand, name, supercategory}}

# Usage:
#   python build_gallery.py \
#     --crops  /path/to/train_gallery_crops \
#     --meta   /path/to/instances_train2019.json \
#     --weights models/vit_sku_best.pt \
#     --out    data/

# The crops directory layout must match what your notebook produced:
#   train_gallery_crops/
#     <sku_id>/
#       img_001.jpg
#       img_002.jpg
#       ...
# """

# import argparse
# import json
# import numpy as np
# import torch
# from pathlib import Path
# from tqdm import tqdm
# from torchvision import transforms
# from PIL import Image
# from utils.vit_model import ViTSKUEncoder

# # ──────────────────────────────────────────────────────────────
# # Args
# # ──────────────────────────────────────────────────────────────
# parser = argparse.ArgumentParser()
# parser.add_argument("--crops",   required=True, help="Path to crop root dir")
# parser.add_argument("--meta",    required=True, help="C:\Users\USER\Downloads\retail_product_checkout\instances_train2019.json")
# parser.add_argument("--weights", default="models/vit_sku_best.pt")
# parser.add_argument("--out",     default="data/")
# parser.add_argument("--embed_dim", type=int, default=128)
# parser.add_argument("--batch",   type=int, default=64)
# args = parser.parse_args()

# # ──────────────────────────────────────────────────────────────
# # Load metadata
# # ──────────────────────────────────────────────────────────────
# with open(args.meta) as f:
#     meta = json.load(f)

# # Build sku_id → {brand, name, supercategory} from RPC categories
# sku_lookup = {}
# for cat in meta["categories"]:
#     sku_id = str(cat["id"])
#     sku_lookup[sku_id] = {
#         "brand":        cat.get("brand",        cat["name"].split("-")[0]),
#         "name":         cat.get("name",         "Unknown"),
#         "supercategory":cat.get("supercategory","Unknown"),
#     }

# # ──────────────────────────────────────────────────────────────
# # Load model
# # ──────────────────────────────────────────────────────────────
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model  = ViTSKUEncoder(embed_dim=args.embed_dim)
# model.load_state_dict(torch.load(args.weights, map_location=device))
# model.eval().to(device)

# tf = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])

# # ──────────────────────────────────────────────────────────────
# # Build gallery
# # ──────────────────────────────────────────────────────────────
# crop_root  = Path(args.crops)
# all_embs   = []
# all_ids    = []

# for sku_dir in tqdm(sorted(crop_root.iterdir()), desc="Building gallery"):
#     if not sku_dir.is_dir():
#         continue
#     sku_id = int(sku_dir.name)
#     imgs   = list(sku_dir.glob("*.jpg")) + list(sku_dir.glob("*.png"))
#     if not imgs:
#         continue

#     batch_tensors = []
#     for img_path in imgs:
#         try:
#             img = Image.open(img_path).convert("RGB")
#             batch_tensors.append(tf(img))
#         except Exception:
#             continue

#     for i in range(0, len(batch_tensors), args.batch):
#         batch = torch.stack(batch_tensors[i:i+args.batch]).to(device)
#         with torch.no_grad():
#             embs = model(batch)   # (B, D) L2-normalised
#         all_embs.extend(embs.cpu().numpy())
#         all_ids.extend([sku_id] * len(batch))

# # ──────────────────────────────────────────────────────────────
# # Save
# # ──────────────────────────────────────────────────────────────
# out_dir = Path(args.out)
# out_dir.mkdir(parents=True, exist_ok=True)

# np.save(out_dir / "sku_gallery_embeddings.npy", np.array(all_embs, dtype=np.float32))
# np.save(out_dir / "sku_gallery_ids.npy",        np.array(all_ids,  dtype=np.int32))

# with open(out_dir / "sku_lookup.json", "w") as f:
#     json.dump(sku_lookup, f, indent=2)

# print(f"\n✅  Gallery saved to {out_dir}")
# print(f"   Embeddings : {len(all_embs)} vectors of dim {args.embed_dim}")
# print(f"   Unique SKUs: {len(set(all_ids))}")


# ============================================================
# CELL 2: Load metadata + build sku_lookup EARLY
# FIX: sku_lookup built here so Cell 8 gallery can use it correctly
# FIX: BRAND_MAP expanded to cover all 200 RPC SKUs
# ============================================================

import json, torch

with open(r"F:\sku_website\data\label_to_cat.json") as f:
    label_to_cat = json.load(f)

with open(r"F:\sku_website\data\sku_lookup.json") as f:
    sku_lookup = json.load(f)

gallery = torch.load(r"F:\sku_website\models\sku_gallery.pt", map_location="cpu")
label_to_meta = gallery.get("label_to_meta", {})

# Check 5 entries
for lbl in list(label_to_meta.keys())[:200]:
    cat_id = label_to_cat.get(str(lbl))
    from_gallery = label_to_meta[lbl]
    from_json    = sku_lookup.get(str(cat_id), {})
    match = from_gallery["name"] == from_json.get("name", "")
    print(f"label {lbl} → cat {cat_id} | name match: {match} | {from_gallery['name']}")

    
# import pandas as pd
# import json


# with open(f"C:\\Users\\USER\\Downloads\\retail_product_checkout\\instances_train2019.json", 'rb') as f:
#     train_data = json.load(f)

# sku_df   = pd.DataFrame(train_data['__raw_Chinese_name_df'])
# anns_df  = pd.DataFrame(train_data['annotations'])
# imgs_df  = pd.DataFrame(train_data['images'])
# img_dict = imgs_df.set_index('id')['file_name'].to_dict()

# cat_ids      = sorted(sku_df['category_id'].unique().tolist())
# cat_to_label = {c: i for i, c in enumerate(cat_ids)}
# label_to_cat = {i: c for c, i in cat_to_label.items()}
# NUM_CLASSES  = len(cat_ids)

# print(f'Unique SKUs (classes): {NUM_CLASSES}')
# print('All SKU names in dataset:')
# print(sku_df[['category_id','name','sku_class']].to_string())

# # ── EXPANDED BRAND_MAP — covers all RPC 200-class SKUs ───────
# # Includes your original map + additional brands found in RPC dataset
# BRAND_MAP = {
#     # Beverages — drinks
#     '百事可乐': 'Pepsi',
#     '可口可乐': 'Coca-Cola',
#     '雪碧': 'Sprite',
#     '芬达': 'Fanta',
#     '农夫山泉': 'Nongfu Spring',
#     '王老吉': 'Wanglaoji',
#     '加多宝': 'Jiaduobao',
#     '怡宝': 'Yibao',
#     '娃哈哈': 'Wahaha',
#     '维他': 'Vita',
#     '茶派': 'Chapai',
#     '优乐美': 'Youlemei',
#     '美涛': 'Meitao',
#     '活力宝': 'Huilibao',
#     '百怡': 'Baiyi',
#     '荣怡': 'Rongyi',
#     # Alcohol
#     '喜力': 'Heineken',
#     '百威': 'Budweiser',
#     '青岛': 'Tsingtao',
#     '雪花': 'Snow Beer',
#     '燕京': 'Yanjing',
#     '珠江': 'Pearl River',
#     # Dairy / Milk
#     '伊利': 'Yili',
#     '蒙牛': 'Mengniu',
#     '光明': 'Bright Dairy',
#     '三元': 'Sanyuan',
#     '银鹭': 'Yinlu',
#     '永和': 'Yonghe',
#     '江中': 'Jiangzhong',
#     # Snacks / Candy
#     '德芙': 'Dove',
#     '士力架': 'Snickers',
#     '好时': 'Hershey',
#     '彩虹糖': 'Skittles',
#     '星爆': 'Starburst',
#     '阿尔卑斯': 'Alps',
#     '熊博士': 'Dr. Bear',
#     '炫迈': 'Stride',
#     '绿箭': 'Extra',
#     '奥利奥': 'Oreo',
#     '百力滋': 'Pocky',
#     '纳宝帝': 'Nabati',
#     '奇多': 'Cheetos',
#     '妙脆角': 'Doritos',
#     '上好佳': 'Oishi',
#     '旺仔': 'Want Want',
#     '洽洽': 'Qiaqia',
#     '脆香米': 'Crispy Rice',
#     '嘉顿': 'Garden',
#     '比巴卜': 'Bibabo',
#     '盼盼': 'Panpan',
#     '达利园': 'Dali Garden',
#     '爱时乐': 'Aishile',
#     '桂力': 'Guili',
#     '车仔': 'Che Zai',
#     '爱乡亲': 'Aixiangqin',
#     '宝鼎': 'Baoding',
#     '庆联': 'Qinglian',
#     '菜园': 'Caiyuan',
#     '鸿泰': 'Hongtai',
#     '今野': 'Jinye',
#     '五谷道场': 'Wugu',
#     # Instant noodles
#     '康师傅': 'Master Kong',
#     '合味道': 'Cup Noodles',
#     '华丰': 'Huafeng',
#     '雀巢': 'Nestle',
#     # Condiments / Seasoning
#     '恒顺': 'Hengshun',
#     '太太乐': 'Taitaile',
#     '家乐': 'Knorr',
#     '味好美': 'McCormick',
#     '海星': 'Haixing',
#     '东古': 'Donggu',
#     '欣和': 'Xin He',
#     '广博': 'Guangbo',
#     # Tissues / Paper
#     '清风': 'Qingfeng',
#     '洁柔': 'Jierou',
#     '相印': 'Xiang Yin',
#     '洁云': 'Jieyun',
#     '舒洁': 'Scotties',
#     '得宝': 'Tempo',
#     '斑布': 'Bambook',
#     # Personal care
#     '高露洁': 'Colgate',
#     '云南白药': 'Yunnan Baiyao',
#     '李施德林': 'Listerine',
#     '蓝月亮': 'Blue Moon',
#     '清扬': 'Clear',
#     '舒亮': 'Shuliang',
#     '舒克': 'Shu Ke',
#     '惠宜': 'Huiyi',
#     '金鱼': 'Goldfish',
#     # Stationery
#     '晨光': 'Morning Glory',
#     '马培德': 'Maped',
#     '东亚': 'Dongya',
#     '米奇': 'Mickey',
#     # Additional RPC brands
#     '统一': 'Uni-President',
#     '乐天': 'Lotte',
#     '日清': 'Nissin',
#     '卫龙': 'Weilong',
#     '三只松鼠': 'Three Squirrels',
#     '良品铺子': 'Bestore',
#     '百草味': 'Be&Cheery',
#     '乐事': "Lay's",
#     '薯愿': 'Oishi Potato',
#     '劲仔': 'Jingzai',
#     '香辣': 'Spicy',
#     '来伊份': 'Lyfen',
#     '大白兔': 'White Rabbit',
#     '徐福记': 'Xu Fu Ji',
#     '金帝': 'Golden Monkey',
#     '费列罗': 'Ferrero',
#     '金莎': 'Ferrero Raffaello',
#     '瑞士莲': 'Lindt',
#     '明治': 'Meiji',
#     '格力高': 'Glico',
#     '卡乐比': 'Calbee',
#     '哈尔滨': 'Harbin Beer',
#     '乌苏': 'Wusu',
#     '钱江': 'Qianjiang',
#     '申江': 'Shenjiang',
#     '特仑苏': 'Terun Su',
#     '纯甄': 'Chunzhen',
#     '安慕希': 'Ambpossible',
#     '君乐宝': 'Junlebao',
#     '菊花': 'Chrysanthemum',
#     '八宝粥': 'Eight Treasure',
#     '老干妈': 'Laoganma',
#     '豆瓣': 'Doubanjiang',
#     '美味鲜': 'Meiweixian',
#     '品品好': 'Pinpinhao',
#     '帮宝适': 'Pampers',
#     'Aji泡芙饼干芒果菠萝味60g':'Aji Puff Biscuits Mango',
#     'MM花生牛奶巧克力豆40g':'M&M S'
# }

# def extract_brand(chinese_name):
#     for cn, en in BRAND_MAP.items():
#         if cn in chinese_name:
#             return en
#     return chinese_name  # FIX: return the Chinese name itself, not 'Unknown'

# sku_df['brand'] = sku_df['name'].apply(extract_brand)
# sku_lookup = sku_df.set_index('category_id')[
#     ['sku_class', 'name', 'brand', 'code']
# ].to_dict(orient='index')

# # Audit: show SKUs whose brand still falls back to Chinese name
# unmapped = sku_df[sku_df['brand'] == sku_df['name']]
# if len(unmapped) > 0:
#     print(f'\nWARNING: {len(unmapped)} SKUs not in BRAND_MAP (using Chinese name as brand):')
#     print(unmapped[['category_id','name','sku_class']].to_string())
# else:
#     print('\nAll SKUs mapped to English brand names.')


# # Save label_to_cat mapping
# with open(f'F:/sku_website/data/label_to_cat.json', 'w') as f:
#     json.dump({str(k): v for k, v in label_to_cat.items()}, f)

# # Save sku_lookup for app.py (built later in the same cell — run after it's defined)
# with open(f'F:/sku_website/data/sku_lookup.json', 'w') as f:
#     json.dump({str(k): v for k, v in sku_lookup.items()}, f)

# print(f"Saved {len(label_to_cat)} label mappings")
# print("Sample:", dict(list(label_to_cat.items())[:3]))