# RAG-Enhanced Sentiment Classifier for Financial Questions

A lightweight yet powerful financial questions classifier that uses a Retrieval-Augmented Generation (RAG) pipeline to summarize and evaluate news headlines for extra context. 
Fine-tuned transformer models on Hugging face datasets are used to classify queries as **Positive**, **Neutral**, or **Negative**.

## Getting started
The a demo code is embedded in a Streamlit app, for instant use and glance. It can be publicly accessed at:

https://finsentimentclassifier.streamlit.app/

A full workflow example is given at "sentanalyst.ipynb" Jupyter notebook. To download and run this project locally follow the following steps:

## Local Set-up Instructions
```
git clone https://gitlab.com/bcucco/rag-enhanced-sentiment-classifier-for-financial-questions.git
cd rag-enhanced-sentiment-classifier-for-financial-questions
pip install -r requirements.txt
streamlit run app.py
```
The Streamlit app will launch in your default browser. You can classify queries like “How is Microsoft stock doing?”. The classification is done using a RAG pipeline to parse recent news for extra context.

## Project structure
```
rag-enhanced-sentiment-classifier/
├── app.py                     # Main streamlit app code.
├── requirements.txt           # Python requirements.
├── sentanalyst.ipynb          # Full workflow Jupyter notebook.
├── README.md                  # This file.
└── models/                    # Optional local model storage (not committed)
```

**NOTE:** Models are hosted on Hugging Face Hub and pulled dynamically in app.py. No key or token required.

## Hosted models
- **bcco/finetuned-128-model**
Model fine-tuned with a maximum of 128 Tokens.
- **bcco/finetuned-256-model**
Model fine-tuned with a maximum of 256 Tokens.

## Features
- 🔍 RAG pipeline for context building with FAISS + Sentence-BERT.

- 📰 Up-to-date financial news via Google News RSS. Easily extendable for other sources and API.

- 🤖 Fine-tuned transformers for finance sentiment analysis. Fine-tuned with Hugging face "financial_phrasebank" dataset.


## License
This project is protected under the terms impose on its license.
See the [LICENSE](LICENSE) file for more details.

![License](https://img.shields.io/badge/license-Custom-lightgrey.svg)


## Authors and Acknowledgments
Developed by Bruno Cucco.

Powered by Hugging Face Transformers, FAISS, and Streamlit.

