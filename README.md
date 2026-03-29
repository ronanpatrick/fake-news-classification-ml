# Classifying Real vs. Fake News: A Machine Learning Approach

**Authors:** Ronan Patrick G. Miralion & Lorenzo Miguel I. Vallefas  

## Project Overview
This project serves as an experimental baseline to evaluate whether traditional machine learning approaches can effectively classify legitimate versus fabricated news using a dataset of 150,000 articles. By investigating foundational data cleaning strategies, vectorization methods, and baseline classification algorithms, this study tests the viability and limits of traditional machine learning for complex natural language tasks.

**Read the Full Report:** For a deep dive into our methodology, variance analysis, and visual proofs, please read our formal academic paper: [`Classifying Real vs. Fake News_ A Machine Learning Approach .pdf`](Classifying%20Real%20vs.%20Fake%20News_%20A%20Machine%20Learning%20Approach%20.pdf).

## Repository Structure & Quick Start
You can view the code here on GitHub, or click the badges below to run the notebooks interactively in Google Colab. The project workflow is divided into three sequential steps:

* **`01_EDA_and_Data_Exploration.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ronanpatrick/fake-news-classification-ml/blob/main/01_EDA_and_Data_Exploration.ipynb)
  * Initial data loading, missing value checks, and visual distribution analysis. 
* **`02_Data_Preparation_and_Vectorization.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ronanpatrick/fake-news-classification-ml/blob/main/02_Data_Preparation_and_Vectorization.ipynb)
  * Custom advanced text-cleaning pipeline (removing noise, standardizing URLs/numbers, lemmatization).
  * Feature engineering evaluation (Bag of Words, TF-IDF, and Word2Vec) using Principal Component Analysis (PCA) to visually confirm class separability.
* **`03_Model_Training_and_Evaluation.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ronanpatrick/fake-news-classification-ml/blob/main/03_Model_Training_and_Evaluation.ipynb)
  * Final classifier training, testing, and performance metrics.
  * Benchmarks Logistic Regression, Linear SVM, and Random Forest models using both TF-IDF and Word2Vec embeddings.
* **`requirements.txt`**
  * List of required Python dependencies (e.g., `pandas`, `scikit-learn`, `nltk`, `matplotlib`).

## Dataset Access & Acknowledgments
Due to the large size of the 150,000-article corpus, the raw datasets and the `training_cleaned_full.csv` file are hosted externally on Kaggle. You can explore and download the complete dataset here: **[https://www.kaggle.com/datasets/ronanpatrick/real-and-fake-news-dataset](https://www.kaggle.com/datasets/ronanpatrick/real-and-fake-news-dataset)**. To run the notebooks locally, download the CSV files and place them in the root directory.

The 150,000-article dataset utilized in this study was provided by the School of Engineering, Computing and Architecture at National University – Dasmariñas for academic purposes. We do not claim ownership of the original text corpus. It is hosted externally strictly to demonstrate the data cleaning and machine learning pipelines developed for this project.

## Methodology
To optimize text classification, we developed a cleaning pipeline that merged article titles with body content to preserve crucial vocabulary. During the exploratory phase, Term Frequency-Inverse Document Frequency (TF-IDF) provided the clearest visual class separation during PCA. In contrast, Word2Vec captured a higher total variance (45.03%) but produced visually overlapping clusters due to its averaging effect on document vectors. 

## Results & Performance
Models were evaluated based on their Accuracy Score on unseen test data.

| Rank | Vectorization | Machine Learning Model | Accuracy Score |
| :--- | :--- | :--- | :--- |
| **1** | **Word2Vec** | **Logistic Regression** | **55.53%** |
| 2 | Word2Vec | SVM (Linear) | 55.51% |
| 3 | TF-IDF | Logistic Regression | 50.90% |
| 4 | TF-IDF | SVM (Linear) | 50.80% |
| 5 | Word2Vec | Random Forest | 49.22% |
| 6 | TF-IDF | Random Forest | 48.64% |

Despite TF-IDF providing better visual separation during PCA, Word2Vec consistently achieved higher empirical accuracy across all models during formal training. Simpler linear models outperformed complex ensemble models, with Logistic Regression achieving the strongest predictive outcome.

## Conclusion
This experiment demonstrates that while traditional machine learning models can capture basic textual patterns, their limited accuracy (peaking at 55.53%) indicates they struggle to confidently distinguish real from fake news. The nuanced complexity of human deception likely requires the more robust, context-aware capabilities provided by deep learning networks (such as Neural Networks or Transformers) for highly reliable detection.
