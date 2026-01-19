# AG News Classification 📰

A comprehensive machine learning project for classifying news articles into 4 categories using both **Traditional ML** and **Deep Learning (DistilBERT)** approaches.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Key Findings](#key-findings)
- [Future Improvements](#future-improvements)


## 🎯 Overview

This project implements a complete NLP pipeline for classifying news articles from the AG News dataset into four categories:
- **World** 🌍
- **Sports** ⚽
- **Business** 💼
- **Sci/Tech** 🔬

The project compares traditional machine learning approaches with state-of-the-art deep learning (DistilBERT) to determine the best classification strategy.

## 📊 Dataset

The AG News dataset consists of news articles from more than 2,000 news sources.

| Split | Samples |
|-------|---------|
| Training | 120,000 |
| Test | 7,600 |

**Class Distribution:** Balanced across all 4 categories (25% each)

## 📁 Project Structure

```
AG-News-ML-Classification/
│
├── AG_News_Classification.ipynb  # Main Jupyter notebook
├── README.md                     # Project documentation
├── data/
│   ├── Train_dataset.jsonl       # Training data
│   └── Test_dataset.jsonl        # Test data
└── requirements.txt              # Dependencies (optional)
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sk-koirala/Ag_news_classification.git
   cd Ag_news_classification
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn wordcloud
   pip install scikit-learn nltk
   pip install torch transformers
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## 🤖 Models Implemented

### Traditional Machine Learning
| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear classifier with L2 regularization |
| **Support Vector Machine (SVM)** | Linear SVM with optimized C parameter |
| **Naive Bayes** | Multinomial NB for text classification |
| **Ensemble Methods** | Voting & Stacking classifiers |

### Deep Learning
| Model | Description |
|-------|-------------|
| **DistilBERT** | Fine-tuned transformer model for sequence classification |

### Feature Engineering
- **TF-IDF Vectorization** (10,000 features, trigrams)
- **Text Preprocessing**: Lowercasing, stopword removal, lemmatization
- **Class weight balancing** for F1 optimization

## 📈 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **DistilBERT** | **0.9412** | **0.9415** | **0.9412** | **0.9413** |
| SVM | 0.9092 | 0.9092 | 0.9092 | 0.9091 |
| Logistic Regression | 0.9086 | 0.9084 | 0.9086 | 0.9084 |
| Ensemble | 0.9084 | 0.9083 | 0.9084 | 0.9082 |
| Naive Bayes | 0.8933 | 0.8928 | 0.8933 | 0.8929 |

### Key Achievements
- ✅ **Best Accuracy:** 94.12% (DistilBERT)
- ✅ **Best F1-Score:** 0.9413 (DistilBERT)
- ✅ **ML Baseline:** 90.92% (SVM)
- ✅ **Improvement over ML:** +3.22%

## 🚀 Usage

### Running the Notebook
```bash
jupyter notebook AG_News_Classification.ipynb
```

### Quick Prediction Example
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Predict
text = "Apple releases new iPhone with groundbreaking camera technology"
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)

labels = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
print(f"Predicted: {labels[prediction.item()]}")
```

## 📊 Visualizations

The notebook includes comprehensive visualizations:
- 📈 Class distribution (Pie charts & Bar charts)
- 📊 Text length distribution analysis
- ☁️ Word clouds for each category
- 🔥 Confusion matrices
- 📉 ROC curves & Precision-Recall curves
- 📊 Model comparison charts
- 📈 Training loss curves

## 🔍 Key Findings

1. **Deep Learning Superiority**: DistilBERT outperforms all traditional ML models by ~3%

2. **SVM is Best ML Model**: Among traditional approaches, SVM achieved the highest performance

3. **Balanced Dataset**: Equal class distribution eliminated class imbalance issues

4. **TF-IDF Effectiveness**: Trigram features with 10K vocabulary provided strong baseline

5. **F1 Optimization**: Class weight balancing improved F1-scores across models

## 🔮 Future Improvements

- [ ] Implement other transformer models (BERT, RoBERTa, XLNet)
- [ ] Add hyperparameter tuning with Optuna
- [ ] Deploy model as REST API
- [ ] Add cross-validation for more robust evaluation
- [ ] Implement model interpretability (LIME, SHAP)
- [ ] Create Streamlit/Gradio web interface

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 👤 Author

**Suyash Koirala**
- GitHub: [@sk-koirala](https://github.com/sk-koirala)

## 🙏 Acknowledgments

- AG News Dataset creators
- Hugging Face for the Transformers library
- scikit-learn developers

---

