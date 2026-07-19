# IDRiD Diabetic Retinopathy Severity Grading

Two notebooks for the same task (Task 1): classifying retinal fundus images
into 5 diabetic retinopathy severity grades (0 = No DR → 4 = Proliferative DR)
on the **IDRiD** dataset.

| Notebook | Purpose |
|---|---|
| `Task_1.ipynb` | Main pipeline. Trains **EfficientNet-B3** with Bayesian HPO (Optuna), evaluates, and runs Grad-CAM error analysis. This is the primary model. |
| `Model-comparisons-task-1.ipynb` | Benchmark notebook. Trains **ResNet-50** and **MobileNetV3-Large** under matching conditions and compares them against the EfficientNet-B3 result from `Task_1.ipynb`. |

Both notebooks are written for **Kaggle** (they auto-detect `/kaggle/input` /
`/kaggle/working`), with `Task_1.ipynb` also able to fall back to a local
`./data/idrid` folder.

## 1. Environment

- Python 3.12, GPU strongly recommended (both notebooks check `torch.cuda.is_available()` and will run on CPU but very slowly).
- Packages:

```bash
pip install torch torchvision numpy pandas opencv-python matplotlib scikit-learn optuna tensorboard
```

## 2. Dataset

Download the **IDRiD "Disease Grading"** subset (fundus images + severity
labels). Each notebook auto-discovers the dataset by scanning for
folders/files containing keywords like `disease`+`grading`, `train`/`test`
image folders, and `train`/`test` label CSVs (matched by filename, so exact
paths don't matter as long as the structure is intact).

- **On Kaggle:** attach the IDRiD Disease Grading dataset to the notebook as
  an input dataset. `Task_1.ipynb` looks under `/kaggle/input`;
  `Model-comparisons-task-1.ipynb` does the same but has **no local
  fallback** — it only works inside a Kaggle-style `/kaggle/input` layout.
- **Locally (Task_1.ipynb only):** place the data under `./data/idrid`
  (train/test image folders + label CSVs anywhere inside, subfolder names
  don't matter). `Model-comparisons-task-1.ipynb` would need `/kaggle/input`
  to exist locally (e.g. symlinked) to run outside Kaggle.

Each label CSV needs an image-name column and a "retinopathy grade" column
(auto-matched by keyword); everything else is derived from that.

## 3. Run order

The two notebooks are **coupled through a checkpoint**, not independent:

1. **Run `Task_1.ipynb` first, top to bottom.**
   - Trains EfficientNet-B3 (two-phase fine-tuning + Optuna HPO, 7 trials).
   - Saves the final checkpoint to `/kaggle/working/checkpoints/best_model.pt`.
   - Also writes metrics, confusion matrices, and Grad-CAM images to `/kaggle/working/results/`.
2. **Make `best_model.pt` available as a Kaggle input to the second notebook.**
   `Model-comparisons-task-1.ipynb` loads the EfficientNet-B3 weights via:
   ```python
   glob.glob("/kaggle/input/**/best_model.pt", recursive=True)
   ```
   i.e. it reads from `/kaggle/input`, not `/kaggle/working`. On Kaggle, do
   this by saving `Task_1.ipynb`'s output as a dataset (or "Add utility
   script/output") and attaching that dataset as input to the comparison
   notebook. Locally, just place `best_model.pt` somewhere under a directory
   you point the glob at.
3. **Run `Model-comparisons-task-1.ipynb` top to bottom.**
   - Trains ResNet-50 and MobileNetV3-Large from scratch (same two-phase
     schedule, 224×224 input vs. EfficientNet-B3's 300×300).
   - Loads the EfficientNet-B3 checkpoint from step 2 for a fair 3-way comparison.
   - Note: the final comparison table (`EFFNET_RESULTS` dict) uses
     **hardcoded reference numbers** from the EfficientNet-B3 run
     (Val QWK 0.7664 / Test QWK 0.6490) rather than recomputing them — if you
     retrain EfficientNet-B3, update these values manually to match, or rely
     on the later cell that re-evaluates the loaded `effnet` checkpoint directly.
   - Saves `comparison_metrics.json` and confusion matrix plots to `/kaggle/working/results/`.

## 4. What each notebook produces

**`Task_1.ipynb`** → `/kaggle/working/results/`
- `effnet_metrics.json` — val/test QWK, macro-F1, AUC
- `cm_effnet_validation.png`, `cm_effnet_test.png` — confusion matrices
- `scatter_effnet_test.png` — predicted vs. true grade scatter
- `error_distances_test.png` — |true − predicted| histogram
- `gradcam/` — Grad-CAM overlays per grade, plus targeted misclassification examples (Grade 1 misses, Grade 3→0 misses)
- `checkpoints/best_model_FINAL.pt` — copy of the best checkpoint

**`Model-comparisons-task-1.ipynb`** → `/kaggle/working/results/`
- `comparison_metrics.json` — val/test QWK, F1, AUC for all three architectures
- `confusion_<model>_<split>.png` — confusion matrices for EfficientNet-B3, ResNet-50, MobileNetV3-Large
- `/kaggle/working/checkpoints/best_resnet50.pt`, `best_mobilenetv3.pt`

## 5. Key config (for reference)

- Seed: 42 throughout, for reproducibility.
- Preprocessing (Task_1 only): black-border removal → resize → black-corner inpainting (fixes a Grad-CAM-discovered artifact) → CLAHE → Ben Graham preprocessing.
- Loss: focal loss with class weights (both notebooks).
- Schedule: Phase 1 (frozen backbone, head only) → Phase 2 (unfreeze last block/stage + head), AdamW, `ReduceLROnPlateau`, early stopping (patience 5).
- HPO (Task_1 only): Optuna TPE over phase LRs, dropout, weight decay, and unfreeze stage; 7 trials at reduced epoch budget, then a full final run with the best params.
