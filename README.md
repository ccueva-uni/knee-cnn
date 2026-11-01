# Knee-CNN: Multilabel Deep Learning for Knee MRI Analysis

**A deep learning framework for automated multilabel classification of knee MRI findings using multilingual clinical reports and sagittal fat-suppressed DICOM sequences.**

## 1. Project Overview

This project develops a deep learning pipeline for the automated identification of multiple knee abnormalities from Magnetic Resonance Imaging (MRI).

The proposed methodology combines clinical text processing, multilingual weak-label extraction, DICOM image preprocessing, convolutional neural networks (CNNs), transfer learning, and model interpretability techniques.

The objective is to investigate the effectiveness of 2D and 2.5D image representations for multilabel classification of knee MRI findings, while addressing class imbalance, multilingual variability, and potential geographic and linguistic biases.

The project uses the RSNA knee MRI dataset and associated clinical reports, integrating text-derived weak labels with a manually annotated gold-standard subset.

## 2. Objectives

- Develop a multilingual rule-based extractor to identify knee abnormalities from clinical reports.
- Generate weak labels for multilabel classification.
- Identify and select appropriate sagittal fat-suppressed (Sagittal FS) DICOM series.
- Convert variable-length DICOM sequences into fixed-shape NumPy tensors.
- Compare CNN architectures trained from scratch and pretrained models.
- Evaluate the impact of 2D versus 2.5D image representations.
- Address class imbalance through weighted loss functions and focal loss.
- Evaluate model performance using macro AUC, precision, recall, and F1-score.
- Investigate model interpretability using Grad-CAM.
- Analyze potential linguistic and geographic biases.

## 3. Dataset and Target Labels

The project uses knee MRI studies and their corresponding clinical reports.
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data

### 3.1. Dataset organization

| Component | Description |
|---|---|
| Clinical reports | 4,407 training reports |
| Gold-standard annotations | 56 manually annotated cases |
| Selected studies | 656 |
| Candidate Sagittal FS series | 750 |
| Training subset | 480 studies |
| Validation subset | 120 studies |
| Gold-standard test subset | 56 studies |

The dataset is organized at three levels:

- **Study:** A complete knee MRI examination, identified by `StudyInstanceUID`.
- **Series:** A collection of images acquired using the same imaging plane and sequence.
- **Slice:** An individual 2D DICOM image belonging to a series.

### 3.2. Multilabel classification

Each study is represented by 12 target labels.

The labels encode the presence or absence of specific abnormalities. The annotation scheme uses three possible states:

| Value | Meaning |
|---|---|
| 1 | Positive finding |
| 0 | Negative finding |
| -1 | Uncertain or unknown finding |

The unknown state is handled explicitly during model training and evaluation.

The classification framework uses 12 independent sigmoid outputs rather than a mutually exclusive multiclass formulation.

## 4. Project Architecture

The complete workflow consists of four main notebooks and two auxiliary notebooks for validating the multilingual text extraction methodology.

### 4.1. Main pipeline

| # | Notebook | Environment | Main output |
|---|---|---|---|
| 01 | `01-rodilla_g03` | Google Colab | `dataset_cnn.csv`, `series_cnn.csv` |
| 02 | `02-preprocesamiento-dicom_g03_kaggle` | Kaggle | `series_elegidas.csv` |
| 03 | `03-cortes_g03_kaggle` | Kaggle | `.npy` images, `indice_npy.csv`, `log_npy.csv` |
| 04 | `04-entrenamiento_g03_kaggle` | Kaggle GPU | PyTorch models, metrics, Grad-CAM |

### 4.2. Auxiliary multilingual validation notebooks

The following notebooks were developed to evaluate and validate the text extraction function before integrating it into the main pipeline.

| Notebook | Purpose |
|---|---|
| `01_modelo_multilingue_roberta.ipynb` | Validation of the multilingual text extraction methodology using a RoBERTa-based language model. |
| `01_modelo_multilingue_biomedico.ipynb` | Validation using a biomedical language model. |

These notebooks are auxiliary experiments. They are not additional stages of the main image-processing and CNN training pipeline.

Their primary purpose is to investigate the reliability of the extraction rules used to generate weak labels from multilingual clinical reports.

## 5. Methodology

### Phase 1. Clinical text processing and weak-label extraction

**Notebook:** `01-rodilla_g03`

Clinical reports are processed to identify relevant findings and generate multilabel annotations.

The main operations include:

1. Text normalization using Unicode NFKC.
2. Identification of linguistic patterns and medical terminology.
3. Application of multilingual rule-based extraction.
4. Generation of 12 weak labels.
5. Validation against the manually annotated gold-standard subset.
6. Analysis of agreement, Cohen's kappa, and performance by language.
7. Construction of the final study-level and series-level datasets.

**Main outputs:**
- `dataset_cnn.csv`
- `series_cnn.csv`

The resulting dataset contains 656 selected studies and 750 candidate Sagittal FS series.

### Phase 2. DICOM series selection

**Notebook:** `02-preprocesamiento-dicom_g03_kaggle`

This stage identifies the appropriate DICOM series for each study.

The selection process considers:

- DICOM series existence and accessibility.
- Number of slices per series.
- Imaging plane and acquisition sequence.
- Fat-suppression characteristics.
- Slice count and candidate-series tie-breaking.
- Laterality identification (right versus left knee).

A total of 656 series are selected, corresponding to one series per study.

**Main output:**
- `series_elegidas.csv`

### Phase 3. DICOM preprocessing and image representation

**Notebook:** `03-cortes_g03_kaggle`

Selected DICOM series are transformed into standardized NumPy arrays suitable for deep learning.

The preprocessing operations include:

1. DICOM reading.
2. Slice ordering.
3. Intensity normalization.
4. Image resizing.
5. Selection of representative slices.
6. Tensor generation.
7. Shape and intensity validation.

Two image representations are considered:

**2D representation**

A single central slice is used:

`[1, H, W]`

**2.5D representation**

Multiple slices distributed along the medial-to-lateral direction are stacked as channels:

`[N, H, W]`

This approach preserves part of the anatomical information along the sagittal acquisition axis while maintaining a 2D CNN-compatible input format.

**Main outputs:**
- `npy/<StudyInstanceUID>.npy`
- `indice_npy.csv`
- `log_npy.csv`

### Phase 4. CNN training and evaluation

**Notebook:** `04-entrenamiento_g03_kaggle`

The processed images and corresponding labels are loaded into a PyTorch dataset.

The modeling stage investigates two main approaches:

**A. CNN architectures trained from scratch**
- 2D CNN.
- 2.5D CNN.

**B. Transfer learning**
- ResNet-50.
- EfficientNet.

The classification head generates 12 independent predictions using sigmoid activation.

The training framework considers:

- `BCEWithLogitsLoss`.
- Class weighting through `pos_weight`.
- Focal loss.
- Data augmentation.
- Study-level data splitting.
- GPU acceleration.

The purpose is to evaluate how different architectures, image representations, and imbalance-handling strategies affect multilabel classification performance.

## 6. Model Evaluation

The final evaluation uses the 56 manually annotated gold-standard cases.

The main evaluation metrics are:

| Metric | Purpose |
|---|---|
| Macro AUC | Overall discrimination performance across labels |
| Per-label AUC | Discrimination performance for each abnormality |
| Precision | Proportion of predicted positives that are correct |
| Recall | Proportion of actual positives correctly identified |
| F1-score | Harmonic mean of precision and recall |

Additional analyses include:

- Confidence intervals using bootstrap resampling.
- Threshold-dependent classification performance.
- Confusion matrices.
- Per-class performance.
- Class imbalance analysis.

## 7. Ablation Study

The ablation study investigates the contribution of different methodological components.

The planned comparisons include:

1. Handling of unknown labels (`-1`).
2. Binary cross-entropy versus focal loss.
3. Sagittal versus coronal image representations, where available.
4. CNN architectures and pretrained backbones.
5. 2D versus 2.5D image representations.

The objective is to identify which components contribute most to the final model performance.

## 8. Model Interpretability: Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is used to visualize the image regions contributing to the model predictions.

The analysis focuses on at least three relevant findings.

The main questions are:

- Does the model focus on anatomically relevant structures?
- Are the activation maps consistent with the expected location of each abnormality?
- Does the model rely on irrelevant image regions or acquisition artifacts?

Grad-CAM provides qualitative evidence to support the interpretation of model predictions.

## 9. Bias and Generalization Analysis

The project investigates potential sources of bias associated with:

- Clinical report language.
- Geographic origin and acquisition center.
- Imaging protocols.
- Class imbalance.
- Variability in DICOM acquisition parameters.

Particular attention is given to linguistic differences and the potential influence of German-language clinical reports on the weak-label extraction process.

The purpose is to assess whether the proposed framework can generalize beyond the specific distribution of the training data.

## 10. Examination Deliverables

The following table identifies where each examination requirement is addressed.

| Deliverable | Notebook | Section |
|---|---|---|
| 1. Weak-label extraction | 01 | Steps 7–8 |
| 2. Multilabel architectures | 04 | 11.2–11.5 |
| 3. Evaluation (AUC, P/R/F1) | 04 | 12 |
| 4. Ablation study | 04 | 13 |
| 5. Grad-CAM | 04 | 14 |
| 6. Class imbalance | 01 and 04 | Steps 2, 9.2; 11.1, 11.4, 12.3 |
| 7. Geographic and linguistic bias | 01, 02, 03 and 04 | Steps 3, 5; 10.1; 10.3; 15 |

## 11. Repository Structure

```text
knee-cnn/
│
├── README.md
│
├── 01_modelo_multilingue_roberta.ipynb
├── 01_modelo_multilingue_biomedico.ipynb
│
├── 01-rodilla_g03.ipynb
├── 02-preprocesamiento-dicom_g03_kaggle.ipynb
├── 03-cortes_g03_kaggle.ipynb
├── 04-entrenamiento_g03_kaggle.ipynb
│
├── extractor.py
│
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── figures/
│   ├── pipeline.png
│   ├── confusion_matrix.png
│   └── gradcam_examples.png
│
└── results/
    └── metrics.csv
```

*Note: The directory structure represents the proposed organization of the repository. Generated datasets, DICOM images, NumPy tensors, and trained weights are maintained separately when their size or licensing conditions make GitHub storage inappropriate.*

## 12. Technologies and Tools

The project uses the following technologies:

| Technology | Application |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| NumPy | Numerical processing and tensor representation |
| Pandas | Dataset management |
| pydicom | DICOM image reading |
| OpenCV | Image preprocessing |
| scikit-learn | Model evaluation |
| Matplotlib | Visualization |
| Grad-CAM | Model interpretability |
| Hugging Face Transformers | Language model experimentation |
| Google Colab | Text processing experiments |
| Kaggle | DICOM preprocessing and GPU training |

## 13. Reproducibility and Data Management

The project separates clinical text processing, DICOM preprocessing, and model training into independent stages.

Intermediate outputs are stored in CSV files and NumPy arrays to facilitate reproducibility and avoid repeating computationally expensive operations.

The dataset and image files are managed through Kaggle datasets, while GitHub is used to maintain the source code, notebooks, documentation, and selected experimental results.

No patient-identifiable information or restricted competition data should be uploaded to a public repository.

## 14. References

- PyTorch: https://pytorch.org/
- Hugging Face: https://huggingface.co/
- pydicom: https://pydicom.github.io/
- scikit-learn: https://scikit-learn.org/
- Grad-CAM: https://arxiv.org/abs/1610.02391
- ResNet: https://arxiv.org/abs/1512.03385
- EfficientNet: https://arxiv.org/abs/1905.11946

---

**Project:** Knee-CNN  
**Repository:** https://github.com/ccueva-uni/knee-cnn

*This project is developed for academic and research purposes. It is not intended to replace professional medical diagnosis or clinical decision-making.*
