# Neural Network SMS Text Classifier

A neural network built with TensorFlow and Keras that classifies SMS messages as "ham" (normal messages) or "spam" (advertisements or unwanted messages). This project is part of the freeCodeCamp Machine Learning with Python certification.

## Overview

- Dataset: SMS Spam Collection, already split into training and validation TSV files
- Text is lowercased, long digit sequences are replaced with a placeholder token, and the pound sign is replaced with a word token, to keep the vocabulary from overfitting to specific phone numbers or currency symbols
- A `TextVectorization` layer turns raw text into integer sequences inside the model, so `predict_message` can take a plain string directly
- Class weights compensate for the dataset containing far more ham than spam messages

## Model

| Component | Details |
|-----------|---------|
| Text vectorization | `TextVectorization`, 10,000-token vocabulary, sequences of length 60 |
| Embedding | 32-dimensional word embeddings |
| Pooling | Global average pooling over the sequence |
| Hidden layer | Dense 32, ReLU, with dropout |
| Output | Dense 1, sigmoid |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Training | Up to 40 epochs with early stopping and class weighting (spam weighted 4x) |

## Usage

```python
predict_message("sale today! to stop texts call 98912460324")
```

Returns a list with the predicted probability of spam and the label, for example `[0.94, 'spam']`.

## Results

The model passes all seven cases in the freeCodeCamp test set, with a validation accuracy typically between 98% and 99%.

## Getting Started

### Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/fcc_sms_text_classification_completed.ipynb)

### Run locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook fcc_sms_text_classification_completed.ipynb
```

The notebook downloads the dataset with `!wget`, which needs a Unix shell. On Windows, use Colab or WSL.

## Project Structure

```
.
├── fcc_sms_text_classification_completed.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── images/
```

## Technologies

Python, TensorFlow, Keras, pandas, NumPy, Matplotlib

## Acknowledgments

Dataset and challenge by [freeCodeCamp](https://www.freecodecamp.org).
