# 🍽️ Zomato Restaurant Reviews & Metadata Clustering Analysis

An Unsupervised Machine Learning and Natural Language Processing (NLP) Capstone Project developed for the **Labmentix Machine Learning Internship Program**.

---

## 📌 Project Overview

The restaurant industry is highly competitive, and customer feedback is a crucial driver of business sustainability and success. This project focuses on analyzing Zomato restaurant reviews and metadata utilizing unsupervised machine learning techniques. 

The primary objective is to group restaurants and customer feedback into distinct, meaningful clusters based on textual sentiment and numerical features without relying on predefined ground-truth labels. This automated segmentation provides restaurant managers with actionable insights into customer preferences, pricing strategies, and operational performance.

---

## 🛠️ Tech Stack & Libraries Used

- **Language:** Python 3.x
- **Data Wrangling & Manipulation:** `Pandas`, `NumPy`
- **Data Visualization:** `Matplotlib`, `Seaborn`
- **Natural Language Processing (NLP):** `NLTK`, `Scikit-Learn` (`TfidfVectorizer`)
- **Machine Learning Algorithms:**
  - Agglomerative Hierarchical Clustering
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  - K-Means Clustering
- **Evaluation Metrics:** Silhouette Score, Davies-Bouldin Index

---

## 🧹 Project Architecture & Workflow

1. **Know Your Data & Wrangling:**
   - Ingested and merged Zomato Restaurant Metadata and Customer Reviews datasets.
   - Handled missing values, removed duplicate entries, and converted data types for consistent analysis.

2. **Text Preprocessing & NLP Pipeline:**
   - Contraction expansion, lowercasing, punctuation, URL, and stopword removal.
   - Word tokenization and Lemmatization.
   - Vectorization using TF-IDF to transform unstructured feedback into numerical feature matrices.

3. **Exploratory Data Analysis (EDA):**
   - Generated 15+ comprehensive visualizations analyzing ratings distribution, cost vs. rating relationships, cuisine trends, and review engagement metrics.
   - Validated observed patterns using statistical hypothesis testing (Pearson correlation, t-tests).

4. **Model Building & Hyperparameter Optimization:**
   - Developed three unsupervised models: **Agglomerative Clustering**, **DBSCAN**, and **K-Means Clustering**.
   - Evaluated models using internal cluster validation metrics: **Silhouette Score** (cohesion/separation) and **Davies-Bouldin Index** (cluster compactness).
   - Tuned hyperparameters ($K$ clusters, linkage criteria, `eps`, `min_samples`) via systematic parameter grid sweeps.

---

## 📊 Key Results & Insights

- **Optimal Segmentation ($K=3$):** K-Means and Agglomerative Hierarchical Clustering effectively segmented restaurants into three distinct performance tiers: *High-Engagement/Top Satisfaction*, *Mid-Tier Standard Dining*, and *Low-Rated/Niche Dining*.
- **Outlier Detection:** DBSCAN successfully isolated anomalous reviews and extreme outlier complaints as noise points.
- **Business Impact:** Automates review categorization without manual tagging, helping management address negative feedback trends early, optimize pricing tiers, and improve customer retention.

---

## 📁 Repository Structure

```text
Labmentix/
│
├── Sample_ML_Submission_Template.ipynb   # Main Jupyter Notebook containing complete code & analysis
├── README.md                             # Project documentation & summary
└── Data/                                 # Dataset files (Zomato Metadata & Reviews)
