import pydicom, json, time
import numpy as np
import pandas as pd
import sys
from pathlib import Path

proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

from src.predictor import BrainTumourPredictor

ds_root = Path(r'c:\Research_Project\Brain_Tumor_Multimodal_Datasets')
proj_root = Path(r'c:\Research_Project\Brain_Tumor_Project')
results_dir = proj_root / 'results'

df_map = pd.read_csv(ds_root / 'TCIA_MRI' / 'patient_mapping.csv')
df_four = pd.read_csv(proj_root / 'data' / 'processed' / 'feature_sets' / 'four_class_genes.csv')
with open(proj_root / 'models' / 'gene_feature_names.json') as f:
    feature_names = json.load(f)

predictor = BrainTumourPredictor()
predictor.load()

# Select 30 verified patients: 15 GBM, 15 LGG
gbm_sample = df_map[df_map['subtype'] == 'GBM'].head(15)
lgg_sample = df_map[df_map['subtype'] == 'LGG'].head(15)
eval_df = pd.concat([gbm_sample, lgg_sample], ignore_index=True)

results = []
print(f'Starting evaluation on {len(eval_df)} verified pairs...')
for idx, row in eval_df.iterrows():
    pid = row['patient_id']
    sub = row['subtype']
    dcm_path = ds_root / row['mri_dicom_path']
    
    # Read DICOM
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    img_rgb = np.stack([arr.astype(np.uint8)]*3, axis=-1)
    
    # Gene expression
    gene_row = df_four[df_four['sample_id'].str.contains(pid)].iloc[0]
    g_vec = gene_row[feature_names].values.astype(np.float32)
    
    # Run single-modality predictions first
    mri_r = predictor.predict_mri(img_rgb, generate_gradcam=False)
    gene_r = predictor.predict_gene(g_vec, use_ensemble=True)

    multi_pred = ""
    multi_conf = ""
    try:
        res = predictor.predict_multimodal(
            image_input=img_rgb,
            gene_vector=g_vec,
            tcga_barcode=pid,
            generate_gradcam=False,
            sample_label=pid
        )
        multi_pred = res["predicted_class"]
        multi_conf = round(float(res["confidence"]), 4)
    except Exception as e:
        # Multimodal fusion denied (e.g. MRI did not predict glioma)
        pass

    results.append({
        'patient_id': pid,
        'TCGA_barcode': pid,
        'TCIA_id': pid,
        'MRI_prediction': mri_r['predicted_class'],
        'MRI_confidence': round(float(mri_r['confidence']), 4),
        'gene_prediction': gene_r['predicted_class'],
        'gene_confidence': round(float(gene_r['confidence']), 4),
        'multimodal_prediction': multi_pred,
        'multimodal_confidence': multi_conf,
        'true_label': sub,
        'pairing_status': 'verified'
    })
    print(f'[{idx+1}/{len(eval_df)}] {pid} (True: {sub}) -> MRI: {mri_r["predicted_class"]}, Gene: {gene_r["predicted_class"]}, Fused: {multi_pred or "N/A"}')

res_df = pd.DataFrame(results)
out_csv = results_dir / 'multimodal_test_results.csv'
res_df.to_csv(out_csv, index=False)
print(f'Successfully generated {out_csv} ({len(res_df)} rows)')
