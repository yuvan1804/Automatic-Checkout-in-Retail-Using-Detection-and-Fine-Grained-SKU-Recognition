# """
# patch_gallery.py
# ================
# Run this ONCE on Kaggle (or wherever your sku_gallery.pt lives) to add
# the missing 'label_to_meta' key.  It reads the existing gallery file,
# rebuilds label_to_meta from sku_lookup (already in your notebook's
# memory), and re-saves the file in-place.

# HOW TO USE
# ----------
# Option A — paste the block below at the END of Cell 8 (replace the old
#            torch.save call).

# Option B — run this as a new cell AFTER Cell 8, with model already loaded.

# Either way, the variables  gallery_labels, label_to_cat, sku_lookup
# must already exist in the notebook's kernel (they do if Cell 8 ran).
# """

# # ══════════════════════════════════════════════════════════════
# # PASTE THIS BLOCK AT THE END OF CELL 8  (replace old torch.save)
# # ══════════════════════════════════════════════════════════════

# # --- Build label_to_meta -------------------------------------------
# # gallery_labels  : 1-D LongTensor of integer labels (one per gallery row)
# # label_to_cat    : dict  label (int) → category_id (int)
# # sku_lookup      : dict  category_id → {sku_class, name, brand, code, …}

# label_to_meta = {}
# for lbl in gallery_labels.cpu().tolist():
#     lbl    = int(lbl)
#     cat_id = label_to_cat[lbl]               # integer RPC category_id
#     info   = sku_lookup.get(cat_id, {})
#     label_to_meta[lbl] = {
#         # 'name'  → English product name shown in the app table
#         # Use sku_df 'name' (Chinese) because that is what the notebook
#         # stores; app.py's SKU_NAME_MAP translates it to English at display.
#         # If you have an English name column, use that instead.
#         'name':      info.get('name',      f'SKU-{cat_id}'),
#         'brand':     info.get('brand',     'Unknown'),
#         # sku_cn is the Chinese SKU name (same as 'name' here)
#         'sku_cn':    info.get('name',      '—'),
#         # sku_class must match VIT_CLASSES / data.yaml supercategory strings
#         # e.g. 'candy', 'drink', 'instant_noodles', etc.
#         'sku_class': info.get('sku_class', 'unknown'),
#     }

# # --- Re-save with the new key -------------------------------------
# torch.save(
#     {
#         'matrix':               gallery_matrix.cpu(),
#         'labels':               gallery_labels.cpu(),
#         'sku_class_to_indices': dict(sku_class_to_indices),
#         'brand_to_indices':     dict(brand_to_indices),
#         'label_to_meta':        label_to_meta,          # ← NEW KEY
#     },
#     f'{WORK_ROOT}/sku_gallery.pt',
# )
# print(f"✅ Saved patched gallery → {WORK_ROOT}/sku_gallery.pt")
# print(f"   {len(label_to_meta)} label→meta entries")

# # Quick sanity check — print first 5 entries
# for k, v in list(label_to_meta.items())[:5]:
#     print(f"  label {k:3d} | brand={v['brand']:20s} | sku_class={v['sku_class']}")


# # ══════════════════════════════════════════════════════════════
# # ALTERNATIVE: if you only have the .pt file and NOT the notebook
# # kernel variables, run this standalone script instead.
# # Set GALLERY_PATH and JSON_PATH to your actual paths.
# # ══════════════════════════════════════════════════════════════

# def rebuild_gallery_standalone():
#     """
#     Standalone rebuilder — use when the notebook kernel is gone.
#     Requires:
#       - sku_gallery.pt  (existing, without label_to_meta)
#       - instances_train2019.json  (RPC dataset annotation file)
#     """
#     import torch, json, pandas as pd, os
#     from collections import defaultdict

#     GALLERY_PATH = r"F:\sku_website\models\sku_gallery.pt"
#     JSON_PATH    = r"F:\sku_website\data\instances_train2019.json"

#     # ── BRAND_MAP (same as notebook Cell 2) ──────────────────
#     BRAND_MAP = {
#         '百事可乐':'Pepsi','可口可乐':'Coca-Cola','雪碧':'Sprite','芬达':'Fanta',
#         '农夫山泉':'Nongfu Spring','王老吉':'Wanglaoji','加多宝':'Jiaduobao',
#         '怡宝':'Yibao','娃哈哈':'Wahaha','维他':'Vita','茶派':'Chapai',
#         '优乐美':'Youlemei','美涛':'Meitao','活力宝':'Huilibao','百怡':'Baiyi',
#         '荣怡':'Rongyi','喜力':'Heineken','百威':'Budweiser','青岛':'Tsingtao',
#         '雪花':'Snow Beer','燕京':'Yanjing','珠江':'Pearl River','伊利':'Yili',
#         '蒙牛':'Mengniu','光明':'Bright Dairy','三元':'Sanyuan','银鹭':'Yinlu',
#         '永和':'Yonghe','江中':'Jiangzhong','德芙':'Dove','士力架':'Snickers',
#         '好时':'Hersheys','彩虹糖':'Skittles','星爆':'Starburst','阿尔卑斯':'Alps',
#         '熊博士':'Dr. Bear','炫迈':'Stride','绿箭':'Extra','奥利奥':'Oreo',
#         '百力滋':'Pocky','纳宝帝':'Nabati','奇多':'Cheetos','妙脆角':'Doritos',
#         '上好佳':'Oishi','旺仔':'Want Want','洽洽':'Qiaqia','脆香米':'Crispy Rice',
#         '嘉顿':'Garden','比巴卜':'Big Babol','盼盼':'Panpan','达利园':'Dali Garden',
#         '爱时乐':'Aishile','桂力':'Guili','车仔':'Che Zai','爱乡亲':'Aixiangqin',
#         '宝鼎':'Baoding','庆联':'Qinglian','菜园':'Caiyuan','鸿泰':'Hongtai',
#         '今野':'Jinye','五谷道场':'Wugu Daochang','康师傅':'Master Kong',
#         '合味道':'Cup Noodles','华丰':'Huafeng','雀巢':'Nestle',
#         'RIO':'RIO','牛栏山':'Niulanshan','KELER':'KELER','QQ星':'QQ Star',
#         '恒顺':'Hengshun','太太乐':'Taitaile','家乐':'Knorr','味好美':'McCormick',
#         '海星':'Haixing','东古':'Donggu','欣和':'Xin He','广博':'Guangbo',
#         '清风':'Qingfeng','洁柔':'Jierou','相印':'Xiang Yin','洁云':'Jieyun',
#         '舒洁':'Scotties','得宝':'Tempo','斑布':'Bambook','高露洁':'Colgate',
#         '云南白药':'Yunnan Baiyao','李施德林':'Listerine','蓝月亮':'Blue Moon',
#         '清扬':'Clear','舒亮':'Shuoliang','舒克':'Shu Ke','惠宜':'Huiyi',
#         '金鱼':'Goldfish','晨光':'Morning Glory','马培德':'Maped',
#         '东亚':'Dongya','米奇':'Mickey','立顿':'Lipton','桂格':'Quaker',
#         '甘源':'Ganyuan','喜多多':'XiDuoDuo','都乐':'Dole','梅林':'Maling',
#         '珠江桥牌':'Pearl River Bridge','古龙':'Gulong','雄鸡标':'Rooster Brand',
#         '舒肤佳':'Safeguard','维达':'Vinda','哈尔滨':'Harbin Beer',
#     }

#     def extract_brand(name):
#         for cn, en in BRAND_MAP.items():
#             if cn in name:
#                 return en
#         return name  # fallback: Chinese name itself

#     # Load RPC annotation JSON
#     print("Loading annotation JSON…")
#     with open(JSON_PATH, 'rb') as f:
#         data = json.load(f)

#     sku_df = pd.DataFrame(data['__raw_Chinese_name_df'])
#     sku_df['brand'] = sku_df['name'].apply(extract_brand)

#     cat_ids      = sorted(sku_df['category_id'].unique().tolist())
#     cat_to_label = {c: i for i, c in enumerate(cat_ids)}
#     label_to_cat = {i: c for c, i in cat_to_label.items()}

#     sku_lookup = sku_df.set_index('category_id')[
#         ['sku_class', 'name', 'brand']
#     ].to_dict(orient='index')

#     # Load existing gallery
#     print("Loading existing gallery…")
#     raw = torch.load(GALLERY_PATH, map_location='cpu')

#     gallery_labels = raw['labels']

#     # Build label_to_meta
#     label_to_meta = {}
#     for lbl in gallery_labels.tolist():
#         lbl    = int(lbl)
#         cat_id = label_to_cat.get(lbl)
#         info   = sku_lookup.get(cat_id, {}) if cat_id is not None else {}
#         label_to_meta[lbl] = {
#             'name':      info.get('name',      f'SKU-{lbl}'),
#             'brand':     info.get('brand',     'Unknown'),
#             'sku_cn':    info.get('name',      '—'),
#             'sku_class': info.get('sku_class', 'unknown'),
#         }

#     # Re-save
#     raw['label_to_meta'] = label_to_meta
#     torch.save(raw, GALLERY_PATH)
#     print(f"✅ Patched gallery saved → {GALLERY_PATH}")
#     print(f"   {len(label_to_meta)} entries")
#     for k, v in list(label_to_meta.items())[:5]:
#         print(f"  label {k:3d} | brand={v['brand']:20s} | sku_class={v['sku_class']}")


# # Uncomment to run standalone (when notebook kernel is unavailable):
# rebuild_gallery_standalone()


import torch, json, pandas as pd
from collections import defaultdict

GALLERY_PATH = r"F:\sku_website\models\sku_gallery.pt"
JSON_PATH    = r"C:\Users\USER\Downloads\retail_product_checkout\instances_train2019.json"

BRAND_MAP = {
    '百事可乐':'Pepsi','可口可乐':'Coca-Cola','雪碧':'Sprite','芬达':'Fanta',
    '农夫山泉':'Nongfu Spring','王老吉':'Wanglaoji','加多宝':'Jiaduobao',
    '怡宝':'Yibao','娃哈哈':'Wahaha','维他':'Vita','茶派':'Chapai',
    '优乐美':'Youlemei','美涛':'Meitao','活力宝':'Huilibao','百怡':'Baiyi',
    '荣怡':'Rongyi','喜力':'Heineken','百威':'Budweiser','青岛':'Tsingtao',
    '雪花':'Snow Beer','燕京':'Yanjing','珠江':'Pearl River','伊利':'Yili',
    '蒙牛':'Mengniu','光明':'Bright Dairy','三元':'Sanyuan','银鹭':'Yinlu',
    '永和':'Yonghe','江中':'Jiangzhong','德芙':'Dove','士力架':'Snickers',
    '好时':'Hersheys','彩虹糖':'Skittles','星爆':'Starburst','阿尔卑斯':'Alps',
    '熊博士':'Dr. Bear','炫迈':'Stride','绿箭':'Extra','奥利奥':'Oreo',
    '百力滋':'Pocky','纳宝帝':'Nabati','奇多':'Cheetos','妙脆角':'Doritos',
    '上好佳':'Oishi','旺仔':'Want Want','洽洽':'Qiaqia','脆香米':'Crispy Rice',
    '嘉顿':'Garden','比巴卜':'Big Babol','盼盼':'Panpan','达利园':'Dali Garden',
    '爱时乐':'Aishile','桂力':'Guili','车仔':'Che Zai','爱乡亲':'Aixiangqin',
    '宝鼎':'Baoding','庆联':'Qinglian','菜园':'Caiyuan','鸿泰':'Hongtai',
    '今野':'Jinye','五谷道场':'Wugu Daochang','康师傅':'Master Kong',
    '合味道':'Cup Noodles','华丰':'Huafeng','雀巢':'Nestle','RIO':'RIO',
    '牛栏山':'Niulanshan','KELER':'KELER','QQ星':'QQ Star','恒顺':'Hengshun',
    '太太乐':'Taitaile','家乐':'Knorr','味好美':'McCormick','海星':'Haixing',
    '东古':'Donggu','欣和':'Xin He','广博':'Guangbo','清风':'Qingfeng',
    '洁柔':'Jierou','相印':'Xiang Yin','洁云':'Jieyun','舒洁':'Scotties',
    '得宝':'Tempo','斑布':'Bambook','高露洁':'Colgate','云南白药':'Yunnan Baiyao',
    '李施德林':'Listerine','蓝月亮':'Blue Moon','清扬':'Clear','舒亮':'Shuoliang',
    '舒克':'Shu Ke','惠宜':'Huiyi','金鱼':'Goldfish','晨光':'Morning Glory',
    '马培德':'Maped','东亚':'Dongya','米奇':'Mickey','立顿':'Lipton',
    '桂格':'Quaker','甘源':'Ganyuan','喜多多':'XiDuoDuo','都乐':'Dole',
    '梅林':'Maling','珠江桥牌':'Pearl River Bridge','古龙':'Gulong',
    '雄鸡标':'Rooster Brand','舒肤佳':'Safeguard','维达':'Vinda','哈尔滨':'Harbin Beer',
}

def extract_brand(name):
    for cn, en in BRAND_MAP.items():
        if cn in name:
            return en
    return name

print("Loading annotation JSON...")
with open(JSON_PATH, 'rb') as f:
    data = json.load(f)

sku_df = pd.DataFrame(data['__raw_Chinese_name_df'])
sku_df['brand'] = sku_df['name'].apply(extract_brand)

cat_ids      = sorted(sku_df['category_id'].unique().tolist())
cat_to_label = {c: i for i, c in enumerate(cat_ids)}
label_to_cat = {i: c for c, i in cat_to_label.items()}
sku_lookup   = sku_df.set_index('category_id')[['sku_class','name','brand']].to_dict(orient='index')

print("Loading existing gallery...")
raw = torch.load(GALLERY_PATH, map_location='cpu')
gallery_labels = raw['labels']

label_to_meta = {}
for lbl in gallery_labels.tolist():
    lbl    = int(lbl)
    cat_id = label_to_cat.get(lbl)
    info   = sku_lookup.get(cat_id, {}) if cat_id is not None else {}
    label_to_meta[lbl] = {
        'name':      info.get('name',      f'SKU-{lbl}'),
        'brand':     info.get('brand',     'Unknown'),
        'sku_cn':    info.get('name',      '—'),
        'sku_class': info.get('sku_class', 'unknown'),
    }

raw['label_to_meta'] = label_to_meta
torch.save(raw, GALLERY_PATH)

print(f"✅ Patched gallery saved → {GALLERY_PATH}")
print(f"   {len(label_to_meta)} label→meta entries")
for k, v in list(label_to_meta.items())[:5]:
    print(f"  label {k:3d} | brand={v['brand']:20s} | sku_class={v['sku_class']}")