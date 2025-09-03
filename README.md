# kenya-constitution-ai-agent
An NLP-powered multilingual AI assistant for exploring the Kenyan Constitution in English and Swahili. Uses Retrieval-Augmented Generation (RAG) with NLP pipelines for accurate, accessible legal Q&amp;A.

## Table of contents


##  Business Context
Kenya’s Constitution is published in both English and Kiswahili, but citizens, students, and policymakers often face challenges in accessing and understanding it.  

This project aims to:
- Classify legal text by language.  
- Retrieve relevant constitutional articles.  
- Support *question answering* and knowledge democratization. 

## Dataset used

All data files are located in the Data/ folder:

| File Name | Source |
|-----------|--------|
| The_Constitution_of_Kenya_2010.pdf | Kenya Law (Official English Version) |
| Kielelezo_Pantanifu_cha_Katiba_ya_Kenya.pdf | Kenya Law (Official Kiswahili Version) |


> All datasets were obtained from publicly available legal sources and are used *strictly for educational and research purposes*.

##  Directory Structure

kenya-constitution-ai/  
├── Data/  
│   ├── The_Constitution_of_Kenya_2010.pdf  
│   ├── Kielelezo_Pantanifu_cha_Katiba_ya_Kenya.pdf  
│   ├── kenya_constitution_structured.csv  
│   └── kenya_constitution_prepared.csv  
├── Notebooks/  
│   └── Kenya_Constitution_AI_Agent.ipynb  
├── Models/  
│   ├── traditional_ml_models.pkl  
│   └── deep_learning_model.h5  
├── Images/  
│   └── (Saved charts, dashboards, and figures)  
├── README.md  
└── .gitignore

##  Data Preparation
- *Dataset size*: 121 legal text samples  
- *Languages*: English & Kiswahili  
- *Preprocessing steps*:
  - Tokenization  
  - Stopword removal  
  - Vectorization (TF-IDF, embeddings for DL)  
- *Splits*: Train (80%) / Test (20%)  

##  Exploratory Data Analysis
- *English texts* contain words like: parliament, rights, constitution, president  
- *Kiswahili texts* contain words like: katiba, bunge, rais, sheria  
- The vocabularies are *highly distinct*, making classification straightforward.

## Visualizations

### Tableau([link](https://public.tableau.com/app/profile/mathews.odongo/viz/Kenya_Constitution/Dashboard1?publish=yes))



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

 ##  Caveat
- Results are perfect, but dataset is *small (121 samples, 25 test samples)*.  
- On *larger/noisier datasets* (slang, code-switching, spelling errors), performance may drop.

##  Findings
- *ML and DL models generalize well*.  
- *Dropout* reduced overfitting in neural nets.  
- DL accuracy was comparable to ML but needed more compute.  
- *ML (Naive Bayes, AdaBoost)* → perfect accuracy.  
- *DL (Neural Nets)* → slightly lower but *scalable*.  
- ML → interpretable (coefficients, feature importance).  
- DL → less interpretable but stronger for contextual tasks.  

---

##  Recommendations
- Save trained models (deep_model.h5, ML pickles).  
- Use ML for *fast classification*.  
- Use DL for *contextual understanding*.  
- Hybrid pipeline → ML (baseline) + DL (semantic tasks).  
- Scale with *Word2Vec, GloVe, BERT, Transformers*.  
- Deploy on *AWS, GCP, Hugging Face*.  
- Retrain with *amendments + case law*.  
- Adopt *active learning* with user feedback.

##  Conclusion

The *Kenya Constitution AI Agent* has progressed from preprocessing and traditional ML models to incorporating *deep learning* for contextual understanding.  
By combining both approaches, the system delivers *accurate, interpretable, and context-rich answers* to legal and constitutional queries.  

This project highlights the potential of *AI in democratizing access to constitutional knowledge, making legal information more accessible to the **public, students, and policymakers*.  

---

## Team
This project was collaboratively developed by:
- Frankline Ondieki ([email](mailto:ondiekifrank021@gmail.com)) | [LinkedIn](https://www.linkedin.com/in/frankline-ondieki-39a61828a/)
- Pacificah Kwamboka Asamba ([email](mailto:sikamboga1@gmail.com)) | [LinkedIn](https://www.linkedin.com/in/pacificah-omboga-42959b83/)
- Diana Macharia ([email](mailto:hellendiana091@gmail.com)) | [LinkedIn](https://www.linkedin.com/in/hellen-diana-njeri)
- Mathews Odongo ([email](mailto:wandera59@gmail.com)) | [LinkedIn](https://www.linkedin.com/in/mathews-odongo-9a2541368?trk=contact-info)
- Nightingale Jeptoo ([email](mailto:nightingalemib@gmail.com)) | [LinkedIn](https://www.linkedin.com/in/jeptoo-nightingale-36131741/)
- Tinah Ngei ([email](mailto:tinahngei@gmail.com)) | [LinkedIn](https://www.linkedin.com/in/tinah-ngei-4b411386/)
