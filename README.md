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
