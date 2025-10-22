# Model Comparison for Crop Disease Detection

**Team Members:** Manisha Chaudhary and Roshan Saud

This repository was developed as part of the curriculum for Virginia Tech's Machine Learning course (CS 4824).

## About the Project

This project compares two deep learning architectures ResNet-50 and EfficientNet-B4 for automated cassava crop disease classification. The dataset used is the Crop Diseases Classification Dataset from Kaggle, which contains images of cassava leaves across five categories: Cassava Bacterial Blight, Cassava Brown Streak, Cassava Green Mottle, Cassava Mosaic, and Healthy. Both models use transfer learning with ImageNet pretrained weights to classify disease types. The dataset is split into 75% training, 15% validation, and 10% test sets using stratified sampling to ensure balanced class representation.

## Installation

```bash
# Clone the repository
git clone https://github.com/mchaudhary22/Model_Comparison_Crop_Disease.git
cd Model_Comparison_Crop_Disease

# Install dependencies
pip install torch torchvision pandas numpy pillow scikit-learn matplotlib seaborn tqdm
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mexwell/crop-diseases-classification) and update the file paths in `dl_model_comparision.py` (lines 22-26).

## Repository Structure

```
Model_Comparison_Crop_Disease/
│
├── CS4824/
│   ├── Processed_CSVs/              # Train/val/test split metadata
│   │   ├── test_df.csv
│   │   ├── train_df.csv
│   │   └── val_df.csv
│   │
│   └── Saved_Models/                # Model outputs and visualizations
│       ├── resnet50/
│       │   ├── resnet50_best.pth
│       │   ├── resnet50_classification_report.csv
│       │   ├── resnet50_confusion_matrix.png
│       │   ├── resnet50_history.csv
│       │   └── test_summary.txt
│       │
│       ├── efficientnet_b4/
│       │   ├── efficientnet_b4_best.pth
│       │   ├── efficientnet_b4_classification_report.csv
│       │   ├── efficientnet_b4_confusion_matrix.png
│       │   ├── efficientnet_b4_history.csv
│       │   └── test_summary.txt
│       │
│       ├── accuracy_comparison.png
│       └── Loss_comparison.png
│
└── dl_model_comparision.py          # Main implementation script
```

## Key Files

- **dl_model_comparision.py** - Complete pipeline for data processing, model training, and evaluation
- **\*_best.pth** - Saved model weights with best validation accuracy
- **\*_history.csv** - Training and validation metrics per epoch
- **\*_classification_report.csv** - Per-class precision, recall, and F1-scores
- **\*_confusion_matrix.png** - Visual confusion matrix
- **test_summary.txt** - Final test accuracy and inference time

## Usage

Run the complete pipeline:

```bash
python dl_model_comparision.py
```

This will:
1. Load and validate data
2. Apply augmentation and create train/val/test splits
3. Train ResNet-50 (25 epochs)
4. Train EfficientNet-B4 (25 epochs)
5. Evaluate both models on test set
6. Generate visualizations and reports

## Results

All results, including trained models, confusion matrices, accuracy plots, and classification reports, are saved in the `Saved_Models/` directory.

## References

Dataset: [Crop Diseases Classification - Kaggle](https://www.kaggle.com/datasets/mexwell/crop-diseases-classification)
