# kenya-constitution-ai-agent
An NLP-powered multilingual AI assistant for exploring the Kenyan Constitution in English and Swahili. Uses Retrieval-Augmented Generation (RAG) with NLP pipelines for accurate, accessible legal Q&amp;A.

##  Business Context
Kenya’s Constitution is published in both English and Kiswahili, but citizens, students, and policymakers often face challenges in accessing and understanding it.  

This project aims to:
- Classify legal text by language.  
- Retrieve relevant constitutional articles.  
- Support *question answering* and knowledge democratization.  

##  Data Preparation
- *Dataset size*: 121 legal text samples  
- *Languages*: English & Kiswahili  
- *Preprocessing steps*:
  - Tokenization  
  - Stopword removal  
  - Vectorization (TF-IDF, embeddings for DL)  
- *Splits*: Train (80%) / Test (20%)  

##  Exploratory Analysis
- *English texts* contain words like: parliament, rights, constitution, president  
- *Kiswahili texts* contain words like: katiba, bunge, rais, sheria  
- The vocabularies are *highly distinct*, making classification straightforward.

##  Models and Performance

We evaluated several models on the task of classifying between *Kiswahili* and *English* texts.  
The models were compared using *Accuracy, F1 Score, and Recall*.

---

###  Traditional ML Models

| Rank | Model                 | Accuracy | F1     | Recall | Notes                          |
|------|-----------------------|----------|--------|--------|--------------------------------|
| 1    | Naive Bayes           | 1.0000   | 1.0000 | 1.0000 | Lightweight, very fast         |
| 1    | AdaBoost (LogReg)     | 1.0000   | 1.0000 | 1.0000 | Strong ensemble model          |
| 2    | Stacking Classifier   | 0.9917   | 0.9917 | 0.9917 | Balanced but slightly lower    |
| 3    | Logistic Regression   | 0.9833   | 0.9833 | 0.9833 | Simple, interpretable baseline |
| 3    | Voting Classifier     | 0.9833   | 0.9833 | 0.9833 | Ensemble of multiple models    |
| 3    | Bagging (LogReg)      | 0.9833   | 0.9833 | 0.9833 | Variance reduction ensemble    |
| 4    | Gradient Boosting     | 0.9667   | 0.9665 | 0.9667 | Slower, prone to overfitting   |
| 5    | Random Forest         | 0.9503   | 0.9331 | 0.9340 | Underperformed due to small data |

###  Deep Learning Models

| Model              | Accuracy | F1     | Recall | Notes                                     |
|--------------------|----------|--------|--------|-------------------------------------------|
| Neural Network (NN)| 0.9800–0.9900 | 0.9800–0.9900 | 0.9800–0.9900 | Required more compute, but scalable to complex tasks |

##  Model Interpretability

Understanding *why* models perform well is just as important as their accuracy.

### 1. Logistic Regression Coefficients
- *Kiswahili indicators* (positive weights):  
  ya, na, wa, katika, katiba, bunge, rais, sheria  
  → Common *function words* and *legal terms* unique to Kiswahili.  

- *English indicators* (negative weights):  
  shall, parliament, constitution, rights, person, president  
  → Formal *legal/constitutional terms* unique to English.  

 *Takeaway: Logistic Regression learns **domain-specific vocabulary* that clearly separates English and Kiswahili.

---

### 2. Classification Report
- *Precision, Recall, F1 = 1.00 (100%)* for both classes.  
- Balanced performance → no bias toward one language.  

 *Takeaway*: The model has perfect reliability on this dataset.

---

### 3. Confusion Matrix
- *English*: 12/12 correctly classified.  
- *Kiswahili*: 13/13 correctly classified.  
- Zero false positives / negatives.  

 *Takeaway*: The model is highly consistent and confident.

---

### 4. Why Simpler Models Work Best
- *High separability of features* → very distinct vocabularies.  
- *Domain-specific dataset* → strong recurring signal words.  
- *Small, balanced dataset* → no class imbalance issues.  

 *Result: Linear models like Naive Bayes and Logistic Regression **excel*, while heavier tree ensembles underperform.

 