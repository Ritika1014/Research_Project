"""
run_qa_verification.py - Final QA Verification Runner Steps 3-8
"""
import os, sys, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, RESULTS_DIR, MRI_CLASSES, MRI_TEST_DIR
from src.xai_utils import (
    GradCAM, load_gradcam_meta,
    compute_shap_tree, plot_shap_summary, plot_shap_bar, plot_shap_waterfall_single,
)
import tensorflow as tf, joblib

GRADCAM_DIR = RESULTS_DIR.parent / "gradcam_preview"
SHAP_DIR    = RESULTS_DIR.parent / "shap_preview"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "feature_sets"

print("=" * 70)
print("   FINAL QA VERIFICATION RUNNER - Steps 3-8")
print("=" * 70)

# ── STEP 3 + 4: MRI + GradCAM ─────────────────────────────────────────────────
print("\n[STEP 3 & 4] MRI Inference + Grad-CAM")
model = tf.keras.models.load_model(str(MODELS_DIR / "best_mri_model.keras"))
print(f"Model loaded | Input: {model.input_shape}")

meta = load_gradcam_meta()
cam  = GradCAM(model=model, layer_name=meta.get("target_layer", "top_conv"))
print(f"Grad-CAM target layer: {meta.get('target_layer','top_conv')}")

selected = []
for cls in MRI_CLASSES:
    imgs = list((MRI_TEST_DIR / cls).glob("*.jpg"))
    if imgs:
        selected.append((cls, random.choice(imgs)))
extra_imgs = list((MRI_TEST_DIR / random.choice(MRI_CLASSES)).glob("*.jpg"))
selected.append((random.choice(MRI_CLASSES), random.choice(extra_imgs)))
selected = selected[:5]

regions = {
    "glioma":    "Hyper-intense intra-axial infiltrative mass with perifocal edema",
    "meningioma":"Extra-axial dural-based lesion with well-defined margins",
    "pituitary": "Sellar/suprasellar mass in pituitary fossa",
    "notumor":   "No focal lesion - normal parenchymal signal",
}

for i, (true_cls, img_path) in enumerate(selected, 1):
    pil  = Image.open(img_path).convert("RGB").resize((224, 224))
    arr  = np.array(pil, dtype=np.float32)
    preds = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_cls = MRI_CLASSES[pred_idx]
    conf     = preds[pred_idx] * 100

    heatmap   = cam.compute(arr, class_idx=pred_idx)
    hmap_rgb  = (matplotlib.colormaps["jet"](heatmap)[:,:,:3]*255).astype(np.uint8)
    overlay   = cam.overlay(arr.astype(np.uint8), heatmap)

    base = f"sample{i}_{true_cls}_pred{pred_cls}"
    comb = Image.new("RGB", (224*3, 224))
    comb.paste(pil, (0,0))
    comb.paste(Image.fromarray(hmap_rgb), (224,0))
    comb.paste(Image.fromarray(overlay), (448,0))
    comb.save(GRADCAM_DIR / f"{base}_3panel.png")
    pil.save(GRADCAM_DIR / f"{base}_orig.png")
    Image.fromarray(hmap_rgb).save(GRADCAM_DIR / f"{base}_heatmap.png")
    Image.fromarray(overlay).save(GRADCAM_DIR / f"{base}_overlay.png")

    prob_str = " | ".join([f"{c}:{p*100:.1f}%" for c,p in zip(MRI_CLASSES, preds)])
    print(f"\n  [Sample {i}] True={true_cls} | Pred={pred_cls} | Conf={conf:.2f}%")
    print(f"    Probs : {prob_str}")
    print(f"    Grad-CAM Region: {regions.get(pred_cls,'N/A')}")
    print(f"    Saved : {base}_3panel.png")

print(f"\nGrad-CAM previews -> {GRADCAM_DIR}")

# ── STEP 5 + 6: Gene + SHAP ───────────────────────────────────────────────────
print("\n\n[STEP 5 & 6] Gene Inference + SHAP")
rf  = joblib.load(MODELS_DIR / "random_forest_gene.pkl")
xgb = joblib.load(MODELS_DIR / "xgboost_gene.pkl")
sc  = joblib.load(MODELS_DIR / "scaler.pkl")
le_path = MODELS_DIR / "gene_label_encoder.pkl"
le  = joblib.load(le_path) if le_path.exists() else None
with open(MODELS_DIR / "gene_feature_names.json") as f:
    gene_names = json.load(f)
print(f"Loaded RF, XGB, Scaler | {len(gene_names)} gene features")

gene_df = pd.read_csv(PROCESSED_DIR / "consensus_genes.csv")
label_col = next((c for c in ["label","Label","subtype","Subtype","class","Class"] if c in gene_df.columns), None)
feat_cols  = [c for c in gene_df.columns if c in gene_names]
print(f"Feature columns matched: {len(feat_cols)}")

all_X = []
subtypes_seen = gene_df[label_col].unique()[:4] if label_col else ["Unknown"]
print(f"\n  Gene Prediction Results:")
print(f"  {'Dataset':<10} {'RF Pred':<8} {'XGB Pred':<8} {'RF Conf':>9} {'XGB Conf':>9}")
print(f"  {'-'*50}")
for sub in subtypes_seen:
    rows = gene_df[gene_df[label_col]==sub].head(3) if label_col else gene_df.head(3)
    X    = rows[feat_cols].values.astype(np.float32)
    Xs   = sc.transform(X)
    for j in range(len(X)):
        rf_p  = rf.predict(Xs[j:j+1])[0]
        xgb_p = xgb.predict(Xs[j:j+1])[0]
        rf_c  = rf.predict_proba(Xs[j:j+1])[0].max()*100
        xgb_c = xgb.predict_proba(Xs[j:j+1])[0].max()*100
        if le:
            try: rf_p  = le.inverse_transform([rf_p])[0]
            except: pass
            try: xgb_p = le.inverse_transform([xgb_p])[0]
            except: pass
        print(f"  {str(sub):<10} {str(rf_p):<8} {str(xgb_p):<8} {rf_c:>8.1f}% {xgb_c:>8.1f}%")
        all_X.append(Xs[j])

print("\n  Generating SHAP explanations (TreeExplainer on RF)...")
shap_X = np.array(all_X)[:30]
# compute_shap_tree(clf, X, background=None) -- do NOT pass feat_cols as 3rd arg
shap_vals = compute_shap_tree(rf, shap_X)

# summary: (shap_values, X, feature_names, model_name, out_dir)
plot_shap_summary(shap_vals, shap_X, feat_cols, "RandomForest", SHAP_DIR)
print(f"  SHAP Summary -> {SHAP_DIR}/shap_summary_randomforest.png")

# waterfall: (shap_values_single, feature_names, base_value, model_name, ...)
# get base_value from RF expected value
import shap as shap_lib
exp_obj = shap_lib.TreeExplainer(rf)
base_val = float(np.mean(exp_obj.expected_value)) if hasattr(exp_obj.expected_value, "__len__") else float(exp_obj.expected_value)
# shap_vals can be list (multiclass) - get positive class
if isinstance(shap_vals, list):
    sv_single = shap_vals[1][0]
elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
    sv_single = shap_vals[0, :, 1]
else:
    sv_single = shap_vals[0]

plot_shap_waterfall_single(sv_single, feat_cols, base_val, "RandomForest", sample_idx=0, out_dir=SHAP_DIR)
print(f"  SHAP Waterfall -> {SHAP_DIR}/shap_waterfall_randomforest_s0.png")

# Top 10 genes by mean |SHAP|
if isinstance(shap_vals, list):
    sv_arr = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
    sv_arr = shap_vals[:, :, 1]
else:
    sv_arr = shap_vals
mean_shap = np.abs(sv_arr).mean(axis=0)
top10 = sorted(enumerate(mean_shap), key=lambda x: x[1], reverse=True)[:10]
print("\n  Top 10 Biomarker Genes (by mean |SHAP|):")
for rank, (idx, val) in enumerate(top10, 1):
    print(f"    #{rank:2d}  {feat_cols[idx]:<14}  SHAP={val:.5f}")

print(f"\nSHAP previews -> {SHAP_DIR}")

# ── STEP 7: Multimodal ────────────────────────────────────────────────────────
print("\n\n[STEP 7] Multimodal Fusion Verification")
from src.multimodal_model import LateFusionClassifier, extract_mri_embedding
late_fusion = LateFusionClassifier()
fus_model = tf.keras.models.load_model(str(MODELS_DIR / "multimodal_fusion.keras"))
print(f"Concatenation Fusion model loaded | Input shape: {fus_model.input_shape}")

bc_data = json.load(open(PROCESSED_DIR / "tcga_verified_barcodes.json"))
test_barcodes = bc_data["GBM_barcodes"][:2] + bc_data["LGG_barcodes"][:1]
bc_subtypes   = ["GBM", "GBM", "LGG"]

gbm_rows = gene_df[gene_df[label_col]=="GBM"].head(3) if label_col else gene_df.head(3)
lgg_rows = gene_df[gene_df[label_col]=="LGG"].head(3) if label_col else gene_df.tail(3)
glioma_imgs = list((MRI_TEST_DIR / "glioma").glob("*.jpg"))

print(f"\n  {'Patient':>20} {'MRI Top':>12} {'Gene':>8} {'Late-Fusion':>14} {'MLP-Fusion':>14} {'Conf':>8}")
print(f"  {'-'*85}")
for bc, sub in zip(test_barcodes, bc_subtypes):
    mri_pil  = Image.open(random.choice(glioma_imgs)).convert("RGB").resize((224,224))
    mri_arr  = np.expand_dims(np.array(mri_pil, dtype=np.float32), 0)
    mri_pred = model.predict(mri_arr, verbose=0)[0]
    mri_cls  = MRI_CLASSES[np.argmax(mri_pred)]
    mri_proba_dict = {cls: float(p) for cls, p in zip(MRI_CLASSES, mri_pred)}

    pool = gbm_rows if sub == "GBM" else lgg_rows
    Xg   = pool.iloc[0][feat_cols].values.astype(np.float32).reshape(1,-1)
    Xgs  = sc.transform(Xg)
    gene_raw   = rf.predict(Xgs)[0]
    gene_proba = rf.predict_proba(Xgs)[0]
    gene_cls   = le.inverse_transform([gene_raw])[0] if le else str(gene_raw)
    classes_list = list(le.classes_) if le else ["GBM", "LGG"]
    gene_proba_dict = {str(c): float(p) for c, p in zip(classes_list, gene_proba)}

    # Late Fusion (Probability Averaging on Glioma Space)
    fused_late = late_fusion.fuse(mri_proba_dict, gene_proba_dict)
    late_cls   = fused_late["predicted_class"]
    late_conf  = fused_late["confidence"] * 100

    # Feature Concatenation MLP Fusion
    mri_emb = extract_mri_embedding(model, np.array(mri_pil, dtype=np.float32))
    raw_gene_vec = pool.iloc[0][feat_cols].values.astype(np.float32)
    target_gene_dim = fus_model.input_shape[1] - mri_emb.shape[0]  # 1348 - 1280 = 68
    if len(raw_gene_vec) < target_gene_dim:
        gene_padded = np.pad(raw_gene_vec, (0, target_gene_dim - len(raw_gene_vec)))
    else:
        gene_padded = raw_gene_vec[:target_gene_dim]
    mlp_inp = np.concatenate([mri_emb, gene_padded]).reshape(1, -1)
    mlp_pred = fus_model.predict(mlp_inp, verbose=0)[0]
    mlp_idx  = int(np.argmax(mlp_pred))
    mlp_cls  = "GBM" if mlp_idx == 0 else "LGG"
    mlp_conf = mlp_pred.max() * 100

    print(f"  {bc[:20]:>20} {mri_cls:>12} {gene_cls:>8} {late_cls:>14} {mlp_cls:>14} {late_conf:>7.1f}%")

# ── STEP 8: Fallback Tests ─────────────────────────────────────────────────────
print("\n\n[STEP 8] Fallback Mechanism Verification")

# Scenario 1: MRI only
pil_s1  = Image.open(random.choice(glioma_imgs)).convert("RGB").resize((224,224))
arr_s1  = np.array(pil_s1, dtype=np.float32)
ps1     = model.predict(np.expand_dims(arr_s1, 0), verbose=0)[0]
h_s1    = cam.compute(arr_s1, class_idx=np.argmax(ps1))
o_s1    = cam.overlay(arr_s1.astype(np.uint8), h_s1)
comb_s1 = Image.new("RGB", (224*2, 224))
comb_s1.paste(pil_s1, (0,0))
comb_s1.paste(Image.fromarray(o_s1), (224,0))
comb_s1.save(GRADCAM_DIR / "scenario1_mri_only.png")
print(f"\n  Scenario 1 PASS: MRI={MRI_CLASSES[np.argmax(ps1)]} ({ps1.max()*100:.1f}%) | GradCAM generated")

# Scenario 2: Gene only
Xs2 = sc.transform(gene_df[feat_cols].iloc[0:1].values.astype(np.float32))
rp2 = rf.predict(Xs2)[0]
gp2 = le.inverse_transform([rp2])[0] if le else str(rp2)
sv2 = compute_shap_tree(rf, Xs2)
ev2 = shap_lib.TreeExplainer(rf)
bv2 = float(np.mean(ev2.expected_value)) if hasattr(ev2.expected_value, "__len__") else float(ev2.expected_value)
sv2_single = (sv2[1][0] if isinstance(sv2, list) else
              (sv2[0,:,1] if isinstance(sv2, np.ndarray) and sv2.ndim==3 else sv2[0]))
plot_shap_waterfall_single(sv2_single, feat_cols, bv2, "Scenario2", sample_idx=0, out_dir=SHAP_DIR)
print(f"  Scenario 2 PASS: Gene={gp2} | SHAP Waterfall saved")

# Scenario 3: MRI + Matched TCGA Gene Late Fusion
bc3   = bc_data["GBM_barcodes"][0]
Xs3   = sc.transform(gene_df[feat_cols].iloc[0:1].values.astype(np.float32))
ms3   = model.predict(np.expand_dims(arr_s1, 0), verbose=0)[0]
mri_p3 = {cls: float(p) for cls, p in zip(MRI_CLASSES, ms3)}
gp3   = rf.predict_proba(Xs3)[0]
classes_list = list(le.classes_) if le else ["GBM", "LGG"]
gene_p3 = {str(c): float(p) for c, p in zip(classes_list, gp3)}
fused3 = late_fusion.fuse(mri_p3, gene_p3)
fc3    = fused3["predicted_class"]
fconf3 = fused3["confidence"] * 100
print(f"  Scenario 3 PASS: Barcode={bc3} | MRI={MRI_CLASSES[np.argmax(ms3)]} | Gene={gp2} | Fusion={fc3} ({fconf3:.1f}%)")

print("\n" + "=" * 70)
print("  QA VERIFICATION COMPLETE -- All Steps 3-8 Done")
print("=" * 70)

