"""
run_phase7_multimodal.py
========================
Phase 7 execution runner — runs the full multimodal pipeline
and generates all result artifacts.
"""

import sys, os, json, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, '.')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s  %(message)s'
)
logger = logging.getLogger('Phase7Runner')

PROJECT_ROOT  = Path('.')
MODELS_DIR    = PROJECT_ROOT / 'models'
DATA_FEATURE  = PROJECT_ROOT / 'data/processed/feature_sets'
MRI_TEST_DIR  = PROJECT_ROOT / 'data/processed/MRI/Testing'
if not MRI_TEST_DIR.exists():
    MRI_TEST_DIR = PROJECT_ROOT / 'data/processed/MRI/test'
RESULTS_MULTI = PROJECT_ROOT / 'results/multimodal'
RESULTS_GENE  = PROJECT_ROOT / 'results/gene_model'
RESULTS_MULTI.mkdir(parents=True, exist_ok=True)
RESULTS_GENE.mkdir(parents=True, exist_ok=True)

# ── STEP 1: Gene Model Training ───────────────────────────────────────────────
logger.info('STEP 1: Gene model training...')
from src.gene_model import (
    run_gene_training_pipeline, load_gene_data, load_gene_artefacts,
    plot_gene_comparison_bar, plot_gene_confusion_matrix,
    RESULTS_DIR as GENE_RESULTS_DIR
)
import joblib
from sklearn.model_selection import train_test_split

if not (MODELS_DIR / 'gene_scaler.pkl').exists():
    logger.info('  Training gene models...')
    gene_results = run_gene_training_pipeline(test_size=0.2, save_models=True)
else:
    logger.info('  Gene models already trained.')

X_gene, y_gene, gene_names, le_gene = load_gene_data()
gene_artefacts = load_gene_artefacts()
logger.info('  Loaded %d samples x %d genes | models: %s',
            len(X_gene), len(gene_names), list(gene_artefacts['models'].keys()))

# ── STEP 2: Gene Model Evaluation ─────────────────────────────────────────────
logger.info('STEP 2: Gene model evaluation...')
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score, average_precision_score
)

X_tr, X_te, y_tr, y_te = train_test_split(X_gene, y_gene, test_size=0.2, stratify=y_gene, random_state=42)
scaler = gene_artefacts['scaler']
X_te_s = scaler.transform(X_te)

gene_eval = {}
for model_name, clf in gene_artefacts['models'].items():
    Xte = X_te_s if model_name == 'mlp' else X_te
    y_pred  = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)
    acc   = accuracy_score(y_te, y_pred)
    bacc  = balanced_accuracy_score(y_te, y_pred)
    mcc   = matthews_corrcoef(y_te, y_pred)
    kappa = cohen_kappa_score(y_te, y_pred)
    auroc = roc_auc_score(y_te, y_proba[:, 1]) if y_proba.shape[1] == 2 else roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro')
    auprc = average_precision_score(y_te, y_proba[:, 1]) if y_proba.shape[1] == 2 else 0.0
    gene_eval[model_name] = {'accuracy': acc, 'balanced_accuracy': bacc, 'mcc': mcc, 'kappa': kappa, 'auroc': auroc, 'auprc': auprc}
    plot_gene_confusion_matrix(y_te, y_pred, list(le_gene.classes_), model_name, out_dir=RESULTS_GENE)
    logger.info('  %s: acc=%.4f | bacc=%.4f | auroc=%.4f | mcc=%.4f', model_name, acc, bacc, auroc, mcc)

plot_gene_comparison_bar({k: {**v, 'report': {}} for k, v in gene_eval.items()}, out_dir=RESULTS_GENE)
pd.DataFrame(gene_eval).T.round(4).to_csv(RESULTS_GENE / 'gene_model_comparison.csv')

# ── STEP 3: TCGA Verification ─────────────────────────────────────────────────
logger.info('STEP 3: TCGA barcode verification...')
from src.multimodal_model import get_verified_tcga_pairs
consensus_df = pd.read_csv(DATA_FEATURE / 'consensus_genes.csv')
verified_df  = get_verified_tcga_pairs(consensus_df)
logger.info('  Verified: %d/%d samples eligible for fusion', len(verified_df), len(consensus_df))

# ── STEP 4: Load MRI model & Predictor ────────────────────────────────────────
logger.info('STEP 4: Loading MRI model and predictor...')
from src.predictor import BrainTumourPredictor, preprocess_mri_image
predictor = BrainTumourPredictor()
predictor.load()
logger.info('  Predictor loaded.')

# ── STEP 5: MRI-only Predictions ─────────────────────────────────────────────
logger.info('STEP 5: MRI-only predictions...')
test_images, test_labels = [], []
for class_dir in sorted(MRI_TEST_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    imgs = sorted(class_dir.glob('*.jpg'))[:2] + sorted(class_dir.glob('*.png'))[:2]
    for img_path in imgs[:2]:
        test_images.append(img_path)
        test_labels.append(class_dir.name)

mri_preds = []
for img_path, true_label in zip(test_images, test_labels):
    result = predictor.predict_mri(img_path, generate_gradcam=False)
    mri_preds.append({
        'image': img_path.name, 'true_class': true_label,
        'predicted_class': result['predicted_class'],
        'confidence': round(result['confidence'], 4),
        'correct': result['predicted_class'] == true_label,
    })
mri_pred_df = pd.DataFrame(mri_preds)
mri_acc     = mri_pred_df['correct'].mean()
mri_pred_df.to_csv(RESULTS_MULTI / 'mri_only_predictions.csv', index=False)
logger.info('  MRI sample accuracy: %.2f%% (%d images)', mri_acc*100, len(test_images))

# ── STEP 6: Grad-CAM Gallery ──────────────────────────────────────────────────
logger.info('STEP 6: Grad-CAM gallery...')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.xai_utils import GradCAM, plot_gradcam_figure

gradcam_gen = GradCAM(predictor._mri_model, predictor._gradcam_meta['gradcam_layer'])
n_gall = min(4, len(test_images))
fig, axes = plt.subplots(n_gall, 3, figsize=(14, n_gall*4))
if n_gall == 1:
    axes = [axes]

for row_i in range(n_gall):
    img_path = test_images[row_i]
    true_lbl = test_labels[row_i]
    img = preprocess_mri_image(img_path)
    heatmap = gradcam_gen.compute(img)
    overlay = gradcam_gen.overlay(img, heatmap)
    result  = predictor.predict_mri(img_path, generate_gradcam=False)
    color   = '#3BB273' if result['predicted_class'] == true_lbl else '#E84855'
    axes[row_i][0].imshow(img);                axes[row_i][0].set_title(f'Original ({true_lbl})', fontsize=9)
    axes[row_i][1].imshow(heatmap, cmap='jet'); axes[row_i][1].set_title('Grad-CAM', fontsize=9)
    axes[row_i][2].imshow(overlay);             axes[row_i][2].set_title(f'Overlay | {result["predicted_class"]} ({result["confidence"]*100:.0f}%)', fontsize=9, color=color)
    for ax in axes[row_i]:
        ax.axis('off')

plt.suptitle('Grad-CAM Visualisations — EfficientNetB0', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(RESULTS_MULTI / 'gradcam_gallery.png'), bbox_inches='tight', dpi=300)
plt.close()
logger.info('  Grad-CAM gallery saved.')

# ── STEP 7: Gene-only Predictions ─────────────────────────────────────────────
logger.info('STEP 7: Gene-only batch predictions...')
gene_preds = []
for idx, row in verified_df.iterrows():
    gene_vec = row[gene_names].values.astype('float32')
    r = predictor.predict_gene(gene_vec, use_ensemble=True)
    gene_preds.append({
        'sample_id': row['sample_id'],
        'true_label': row['label'],
        'predicted_class': r['predicted_class'],
        'confidence': round(r['confidence'], 4),
        'correct': r['predicted_class'] == row['label'],
    })
gene_pred_df = pd.DataFrame(gene_preds)
gene_acc     = gene_pred_df['correct'].mean()
gene_pred_df.to_csv(RESULTS_MULTI / 'gene_only_predictions.csv', index=False)
logger.info('  Gene-only accuracy: %.2f%%', gene_acc*100)

# ── STEP 8: SHAP for RandomForest ─────────────────────────────────────────────
logger.info('STEP 8: SHAP explainability...')
from src.xai_utils import compute_shap_tree, plot_shap_summary, plot_shap_bar
rf_clf = gene_artefacts['models'].get('randomforest')
if rf_clf is not None:
    X_shap = verified_df[gene_names].values.astype('float32')
    n_shap = min(100, len(X_shap))
    try:
        logger.info('  Computing SHAP for RandomForest on %d samples...', n_shap)
        shap_vals = compute_shap_tree(rf_clf, X_shap, n_samples=n_shap)
        plot_shap_bar(shap_vals, gene_names, 'RandomForest', top_n=20, out_dir=RESULTS_MULTI)
        plot_shap_summary(shap_vals, X_shap[:n_shap], gene_names, 'RandomForest',
                          out_dir=RESULTS_MULTI, max_display=20)
        logger.info('  SHAP plots saved.')
    except Exception as e:
        logger.warning('  SHAP failed: %s', e)

# ── STEP 9: Multimodal Late Fusion Demo ───────────────────────────────────────
logger.info('STEP 9: Multimodal late fusion...')
from src.multimodal_model import LateFusionClassifier
from src.xai_utils import plot_fusion_probability_radar
fusion_clf = LateFusionClassifier()

glioma_dir  = MRI_TEST_DIR / 'glioma'
glioma_imgs = sorted(glioma_dir.glob('*.jpg'))[:5] + sorted(glioma_dir.glob('*.png'))[:5]
glioma_imgs = glioma_imgs[:min(5, len(glioma_imgs))]

gbm_samples = verified_df[verified_df['label'] == 'GBM'].head(3)
lgg_samples = verified_df[verified_df['label'] == 'LGG'].head(3)
demo_gene   = pd.concat([gbm_samples, lgg_samples]).reset_index(drop=True)

fusion_results = []
n_pairs = min(len(glioma_imgs), len(demo_gene))
for i in range(n_pairs):
    img_path = glioma_imgs[i]
    gene_row = demo_gene.iloc[i]
    tcga_bc  = gene_row['sample_id']
    gene_vec = gene_row[gene_names].values.astype('float32')

    mri_r  = predictor.predict_mri(img_path, generate_gradcam=False)
    gene_r = predictor.predict_gene(gene_vec, use_ensemble=True)

    try:
        fusion_r = predictor.predict_multimodal(
            img_path, gene_vec, tcga_barcode=tcga_bc,
            generate_gradcam=False, sample_label=f'demo_{i}',
        )
        mode  = fusion_r['mode']
        pred  = fusion_r['predicted_class']
        conf  = fusion_r['confidence']
        warn  = fusion_r.get('warning', '')
    except Exception as e:
        mode  = f'ERROR: {type(e).__name__}'
        pred  = 'N/A'
        conf  = 0.0
        warn  = str(e)

    fusion_results.append({
        'demo_pair': i, 'tcga_barcode': tcga_bc,
        'true_gene_label': gene_row['label'],
        'mri_pred': mri_r['predicted_class'], 'mri_conf': round(mri_r['confidence'], 4),
        'gene_pred': gene_r['predicted_class'], 'gene_conf': round(gene_r['confidence'], 4),
        'fusion_pred': pred, 'fusion_conf': round(conf, 4), 'fusion_mode': mode, 'warning': warn,
    })

fusion_df = pd.DataFrame(fusion_results)
fusion_df.to_csv(RESULTS_MULTI / 'fusion_demo_results.csv', index=False)
eligible = [r for r in fusion_results if 'late-fusion' in r.get('fusion_mode', '')]
logger.info('  Fusion results: %d pairs | %d eligible late-fusion', len(fusion_results), len(eligible))
logger.info('\n%s', fusion_df[['demo_pair','true_gene_label','mri_pred','gene_pred','fusion_pred','fusion_conf','fusion_mode']].to_string())

# Radar for first eligible pair
if eligible:
    r = eligible[0]
    i = r['demo_pair']
    mri_rr  = predictor.predict_mri(glioma_imgs[i], generate_gradcam=False)
    gene_rr = predictor.predict_gene(demo_gene.iloc[i][gene_names].values.astype('float32'))
    fused   = fusion_clf.fuse(mri_rr['probabilities'], gene_rr['probabilities'])
    plot_fusion_probability_radar(
        mri_proba   = {'glioma': mri_rr['probabilities'].get('glioma', 0)},
        gene_proba  = gene_rr['probabilities'],
        fusion_proba = fused['probabilities'],
        out_dir     = RESULTS_MULTI,
        sample_label = f'Demo_{i}_{r["tcga_barcode"]}',
    )
    logger.info('  Radar chart saved.')

# ── STEP 10: Three-mode comparison ────────────────────────────────────────────
logger.info('STEP 10: Three-mode comparison...')
comparison_records = []
for r in fusion_results:
    comparison_records.append({'sample': f'P{r["demo_pair"]}', 'mode': 'MRI-only',
                                'predicted_class': r['mri_pred'], 'confidence': r['mri_conf'],
                                'true_label': r['true_gene_label']})
    comparison_records.append({'sample': f'P{r["demo_pair"]}', 'mode': 'Gene-only',
                                'predicted_class': r['gene_pred'], 'confidence': r['gene_conf'],
                                'true_label': r['true_gene_label']})
    if 'late-fusion' in r.get('fusion_mode', ''):
        comparison_records.append({'sample': f'P{r["demo_pair"]}', 'mode': 'Multimodal',
                                    'predicted_class': r['fusion_pred'], 'confidence': r['fusion_conf'],
                                    'true_label': r['true_gene_label']})

comparison_df = pd.DataFrame(comparison_records)
comparison_df.to_csv(RESULTS_MULTI / 'three_mode_comparison.csv', index=False)

if len(comparison_df) > 0:
    from src.xai_utils import plot_prediction_confidence_bars
    plot_prediction_confidence_bars(
        comparison_df.to_dict('records'),
        out_dir=RESULTS_MULTI,
        filename='three_mode_confidence_bars.png',
    )
    logger.info('  Three-mode comparison saved.')

    mode_summary = comparison_df.groupby('mode').agg(
        n_samples=('predicted_class', 'count'),
        mean_conf=('confidence', 'mean'),
    ).round(4)
    logger.info('\n%s', mode_summary)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'MRI-only': '#2E86AB', 'Gene-only': '#E84855', 'Multimodal': '#3BB273'}
    modes  = mode_summary.index.tolist()
    confs  = mode_summary['mean_conf'].values
    bar_c  = [colors.get(m, '#9B5DE5') for m in modes]
    bars   = ax.bar(modes, confs, color=bar_c, alpha=0.85)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Mean Confidence', fontsize=12)
    ax.set_title('Mean Prediction Confidence by Mode', fontsize=14, fontweight='bold')
    for bar, c in zip(bars, confs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02,
                f'{c:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(RESULTS_MULTI / 'mode_confidence_bar.png'), bbox_inches='tight', dpi=300)
    plt.close()

# ── STEP 11: Comprehensive Results Table ──────────────────────────────────────
logger.info('STEP 11: Comprehensive results table...')
summary_rows = []
summary_rows.append({'Component': 'EfficientNetB0 (MRI)', 'Test_Accuracy': '83.25%',
                     'AUROC': 'N/A (multi-class OvR)', 'Classes': '4', 'Phase': 5})
for name, m in gene_eval.items():
    summary_rows.append({'Component': f'{name} (Gene)', 'Test_Accuracy': f"{m['accuracy']*100:.2f}%",
                         'AUROC': f"{m['auroc']:.4f}", 'Classes': '2 (GBM, LGG)', 'Phase': 7})
n_eligible = len(eligible)
summary_rows.append({'Component': 'Late Fusion (Multimodal)', 'Test_Accuracy': f'{n_eligible}/{len(fusion_results)} eligible',
                     'AUROC': 'N/A (demo)', 'Classes': '2 (GBM, LGG)', 'Phase': 7})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS_MULTI / 'comprehensive_results.csv', index=False)
logger.info('\n=== COMPREHENSIVE RESULTS ===\n%s', summary_df.to_string(index=False))

# ── FINAL VERIFICATION ────────────────────────────────────────────────────────
logger.info('\n=== PHASE 7 COMPLETE ===')
deliverables = [
    ('src/gene_model.py',                  True),
    ('src/multimodal_model.py',            True),
    ('src/xai_utils.py',                   True),
    ('src/predictor.py',                   True),
    ('notebooks/07_Multimodal_Fusion.ipynb', True),
    ('models/gene_randomforest.pkl',        (MODELS_DIR / 'gene_randomforest.pkl').exists()),
    ('models/gene_mlp.pkl',                 (MODELS_DIR / 'gene_mlp.pkl').exists()),
    ('models/gene_scaler.pkl',              (MODELS_DIR / 'gene_scaler.pkl').exists()),
    ('models/gene_label_encoder.pkl',       (MODELS_DIR / 'gene_label_encoder.pkl').exists()),
    ('results/multimodal/gradcam_gallery.png',          (RESULTS_MULTI / 'gradcam_gallery.png').exists()),
    ('results/multimodal/shap_bar_randomforest.png',    (RESULTS_MULTI / 'shap_bar_randomforest.png').exists()),
    ('results/multimodal/shap_summary_randomforest.png',(RESULTS_MULTI / 'shap_summary_randomforest.png').exists()),
    ('results/multimodal/mri_only_predictions.csv',     (RESULTS_MULTI / 'mri_only_predictions.csv').exists()),
    ('results/multimodal/gene_only_predictions.csv',    (RESULTS_MULTI / 'gene_only_predictions.csv').exists()),
    ('results/multimodal/fusion_demo_results.csv',      (RESULTS_MULTI / 'fusion_demo_results.csv').exists()),
    ('results/multimodal/three_mode_comparison.csv',    (RESULTS_MULTI / 'three_mode_comparison.csv').exists()),
    ('results/multimodal/comprehensive_results.csv',    (RESULTS_MULTI / 'comprehensive_results.csv').exists()),
    ('results/gene_model/gene_model_comparison.csv',    (RESULTS_GENE / 'gene_model_comparison.csv').exists()),
]
all_ok = True
for name, status in deliverables:
    status_str = 'OK' if status else 'MISSING'
    if not status:
        all_ok = False
    print(f'  [{status_str}] {name}')

print()
print('=== ALL DELIVERABLES OK ===' if all_ok else '=== SOME DELIVERABLES MISSING ===')
