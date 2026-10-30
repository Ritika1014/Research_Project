import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

from src.gene_model import (
    run_gene_training_pipeline, load_gene_data,
    plot_gene_confusion_matrix, plot_gene_comparison_bar,
    RESULTS_DIR
)
from sklearn.model_selection import train_test_split
import numpy as np, joblib

# Train all gene models
results = run_gene_training_pipeline(test_size=0.2, save_models=True)
print('\n=== GENE MODEL RESULTS ===')
for name, m in results.items():
    acc  = m['accuracy']
    bacc = m['balanced_accuracy']
    auroc = m['auroc']
    mcc   = m['mcc']
    cv    = m['cv_mean']
    print(f'  {name}: acc={acc:.4f} | bacc={bacc:.4f} | auroc={auroc:.4f} | mcc={mcc:.4f} | cv={cv:.4f}')

# Save comparison figure
plot_gene_comparison_bar(results, out_dir=RESULTS_DIR)

# Confusion matrices for each model
X, y, genes, le = load_gene_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = joblib.load('models/gene_scaler.pkl')
X_test_s = scaler.transform(X_test)

for candidate in ['randomforest', 'xgboost', 'mlp']:
    import os
    p = 'models/gene_' + candidate + '.pkl'
    if os.path.exists(p):
        clf = joblib.load(p)
        Xte = X_test_s if candidate == 'mlp' else X_test
        y_pred = clf.predict(Xte)
        plot_gene_confusion_matrix(y_test, y_pred, list(le.classes_), candidate, out_dir=RESULTS_DIR)

print('\nDone. Models and figures saved.')
