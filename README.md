# Time-Aware Explainable Machine Learning for Early At-Risk Student Prediction on OULAD

> *Phát hiện sớm sinh viên có nguy cơ học tập kém bằng học máy có khả năng giải thích*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-green.svg)](LICENSE)
[![Data: CC-BY 4.0](https://img.shields.io/badge/data-CC--BY%204.0-lightgrey.svg)](https://analyse.kmi.open.ac.uk/open_dataset)
[![Status: Early Development](https://img.shields.io/badge/status-early%20development-orange.svg)](#10-project-status)

A reproducible machine-learning pipeline that predicts at-risk students **early** (at multiple points across a course), **explains** every prediction with SHAP and LIME, and **quantifies the stability** of those explanations — all on the public Open University Learning Analytics Dataset (OULAD). This repository accompanies the DSP391m Data Science Capstone Project at FPT University.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Research Questions](#2-research-questions)
3. [Contributions](#3-contributions)
4. [Dataset](#4-dataset)
5. [Repository Structure](#5-repository-structure)
6. [Methodology](#6-methodology)
7. [Getting Started](#7-getting-started)
8. [Reproducibility](#8-reproducibility)
9. [Team & Responsibilities](#9-team--responsibilities)
10. [Project Status](#10-project-status)
11. [Citation](#11-citation)
12. [References](#12-references)
13. [License](#13-license)

---

## 1. Motivation

Failure and drop-out remain persistent problems in Virtual Learning Environments, while the behavioural data captured by Learning Management Systems is largely under-used for timely intervention. Existing predictive models exhibit three recurring limitations:

- **Opacity.** The strongest models behave as black boxes, so instructors cannot see *why* a student is flagged and therefore cannot act with confidence.
- **Lateness.** Most studies predict at end-of-course, when intervention is no longer effective.
- **Class imbalance.** In the literature the at-risk class is usually a minority that models drift away from. Under this project's label mapping ({Fail, Withdrawn} vs {Pass, Distinction}) it is in fact a slight majority on OULAD — 52.8% of enrolments, imbalance ratio 1.12 — so imbalance handling is studied here as a controlled robustness question (RQ3), not as rescuing a rare class.

A concept-centric review of 27 representative papers (2019–2026) shows that *time-aware prediction*, *explainable AI*, and *imbalance handling* have each been studied in isolation but are rarely integrated — and never simultaneously on OULAD. This project targets exactly that empty cell.

## 2. Research Questions

| ID  | Question |
| --- | --- |
| **RQ1** | At different course-progress checkpoints (10–100% of course length), which algorithm gives the best at-risk prediction on OULAD, and how early can a prediction be considered reliable? |
| **RQ2** | How consistent are the explanations produced by SHAP and LIME for the same model, and how does their stability change across time and across imbalance-handling strategies? |
| **RQ3** | How does imbalance handling (SMOTE / ADASYN / class weighting) affect both predictive accuracy and explanation quality? |

## 3. Contributions

1. **An integrated time-aware XAI framework** that couples checkpoint-based prediction with SHAP/LIME explanations at each checkpoint — not previously done end-to-end on OULAD.
2. **An extended OULAD benchmark** adding ensemble methods (Random Forest, XGBoost, LightGBM) to the comparison that Tomasevic et al. (2020) did not include.
3. **A quantitative explanation-stability metric** (Jaccard top-*k* agreement and standard deviation of feature importance across seeds), addressing the qualitative-only gap noted in prior reviews.
4. **An optional instructor dashboard** that turns predictions into a practical early-warning tool.

## 4. Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)** (Kuzilek et al., 2017).

| Property | Value |
| --- | --- |
| Enrolments (student × module-presentation) | 32,593 (28,785 distinct students) |
| Module-presentations | 22 |
| Relational tables | 7 |
| Feature groups | Demographics · Engagement (VLE clickstream) · Performance (assessments) |
| Target | `final_result` mapped to a binary label: **at-risk** = {Fail, Withdrawn}, **not-at-risk** = {Pass, Distinction} |
| License | CC-BY 4.0 (anonymised at source) |
| Source | <https://analyse.kmi.open.ac.uk/open_dataset> · [Kaggle mirror](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad) |

> Note: The raw CSVs are tracked directly in this repository's `Data/` directory using **Git LFS** due to large file sizes (e.g., `studentVle.csv`).

## 5. Repository Structure

The current repository structure reflects the initial phase of the project (Data Understanding & EDA):

```text
time-aware-xai-oulad/
├── Code/
│   └── notebook/
│       ├── 00_data_understanding.ipynb    # Initial EDA and data exploration
│       └── schema_survey.ipynb            # Schema analysis of OULAD
├── Data/                                  # OULAD dataset files (tracked via Git LFS)
│   ├── assessments.csv
│   ├── courses.csv
│   ├── studentAssessment.csv
│   ├── studentInfo.csv
│   ├── studentRegistration.csv
│   ├── studentVle.csv
│   └── vle.csv
├── Others/
│   └── docs/                              # Project documentation
├── Reports/
│   └── data_understanding/                # Reports generated during the EDA phase
└── README.md
```

*(Note: As the project progresses, this structure will expand to include modeling, evaluation, dashboard, etc., following the CRISP-DM lifecycle.)*

## 6. Methodology

The workflow follows the **CRISP-DM** lifecycle and is organised into six phases, each producing an independently assessable deliverable.

| Phase | Focus | Status |
| --- | --- | --- |
| **1. Data Preparation** | Merge the 7 OULAD tables; explore data schemas and understand distributions. | **In Progress** |
| **2. Benchmarking** | Compare LR, RF, XGBoost, LightGBM, ANN. | *Planned* |
| **3. Time-Aware Prediction** | Re-run the benchmark at each checkpoint. | *Planned* |
| **4. Imbalance Handling** | Compare resampling strategies. | *Planned* |
| **5. XAI Layer** | SHAP and LIME evaluation. | *Planned* |
| **6. Dashboard (optional)** | Instructor early-warning UI. | *Planned* |

## 7. Getting Started

### Prerequisites

- Python 3.10 or later
- `git`
- **Git LFS** (Required to download the dataset files)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/HE181429/oulad-student-risk-prediction.git
cd oulad-student-risk-prediction

# 2. Fetch Large Files (Data)
git lfs pull

# 3. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 4. Install dependencies (e.g., Jupyter, Pandas)
pip install jupyter pandas matplotlib seaborn
```

### Reproduce the EDA

To view the current progress on data understanding:
```bash
jupyter notebook Code/notebook/schema_survey.ipynb
```

## 8. Reproducibility

This project is designed so that an external reader can reproduce every result:

- **Data provenance**: Original OULAD tables are available directly in the `Data/` folder and version-controlled via Git LFS.
- **Notebooks**: Execute top-to-bottom without manual intervention.
- **Deterministic seeds**: To be implemented as `RANDOM_SEED = 42` in future modeling stages.

## 9. Team & Responsibilities

Group 1, DSP391m — FPT University. Supervisor: **Nguyễn Thị Hoàng Yến**.

| Member | Role | Responsibility |
| --- | --- | --- |
| Khoa | Methodology Lead | Predictive modelling, time-aware prediction (Themes 1 & 2) |
| Bình | XAI Lead | Explainability (SHAP/LIME), explanation stability (Theme 3) |
| Đức | Modeling Lead | Model development, class-imbalance handling (Themes 1 & 4) |
| Phúc | Implementation Lead | Data pipeline, experiment harness, evaluation |
| Sơn | Literature Review Lead | Introduction, literature review, references |
| An | Backend & Dashboard Lead | Model packaging, Streamlit dashboard (Phase 6a) |

## 10. Project Status

**Phase 1 — Data Understanding & Preparation: In Progress.** 
The project has been initialized. The OULAD dataset has been successfully collected and committed to the `Data/` directory using Git LFS. Initial schema surveys and data understanding notebooks (`00_data_understanding.ipynb`, `schema_survey.ipynb`) have been created in `Code/notebook/` and are currently being developed. 

Subsequent phases (Modeling, Benchmarking, XAI) have not yet started and their respective code/scripts will be added in future updates as the project advances.

## 11. Citation

If you use this work, please cite it as:

```bibtex
@misc{group1_2026_timeaware_xai_oulad,
  title        = {Time-Aware Explainable Machine Learning for Early
                  At-Risk Student Prediction on OULAD},
  author       = {S{\o}n and Khoa and An and {\DJ}{\'u}c and Ph{\'u}c and B{\`i}nh},
  howpublished = {DSP391m Data Science Capstone Project, FPT University},
  year         = {2026},
  note         = {Supervisor: Nguy{\~{\^e}}n Th{\d{i}} Ho{\`a}ng Y{\'{\^e}}n}
}
```

## 12. References

1. N. Tomasevic, N. Gvozdenovic, S. Vranes. *An overview and comparison of supervised data mining techniques for student exam performance prediction.* Computers & Education, 2020.
2. M. Adnan et al. *Predicting at-risk students at different percentages of course length for early intervention.* IEEE Access, 2021.
3. S. Gunasekara, M. Saarela. *Explainable AI in Education: Techniques and Qualitative Assessment.* Applied Sciences, vol. 15, no. 3, art. 1239, 2025.
4. H. Alamri, B. Alharbi. *Explainable Student Performance Prediction Models: A Systematic Review.* IEEE Access, 2021.
5. J. Kuzilek, M. Hlosta, Z. Zdrahal. *Open University Learning Analytics Dataset (OULAD).* Scientific Data, 2017.
6. S. Lundberg, S.-I. Lee. *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS, 2017.
7. M. Ribeiro, S. Singh, C. Guestrin. *"Why Should I Trust You?" Explaining the Predictions of Any Classifier (LIME).* KDD, 2016.
8. N. V. Chawla et al. *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 2002.
9. J. Webster, R. T. Watson. *Analyzing the past to prepare for the future: Writing a literature review.* MIS Quarterly, 2002.

## 13. License

- **Code** is released under the [MIT License](LICENSE).
- **The OULAD dataset** is the property of The Open University and is distributed separately under [CC-BY 4.0](https://analyse.kmi.open.ac.uk/open_dataset); it has been mirrored in this repository for educational purposes under the same license terms.
