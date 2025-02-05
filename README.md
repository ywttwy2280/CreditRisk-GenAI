# CreditRiskModeling-GenAI
A DEMO for modeling Credit Risk with GenAI(Generative AI) Empowered
Overview
This repository contains the implementation of a Credit Risk Model designed to predict the Probability of Default (PD) for borrowers using machine learning techniques. The model integrates GenAI (Generative AI) for feature engineering and model diagnostics to improve accuracy and provide insights into the factors driving credit risk.

##Key Features:

Exploratory Data Analysis (EDA) with GenAI: Use of GenAI for understanding all features's business sense and correlations between predictors

Feature Engineering with GenAI: Use of GenAI for exploring advanced features to enhance model performance.

Modeling: Implementation of various machine learning models to predict the Probability of Default(PD), including Logistic Regression, Gradient Boosting, and Neural Networks.
GenAI Empowered Model Diagnosis: Use of GenAI for evaluating the performance of all ML models

Reporting:Use of GenAI for drafting analysis reports in pdf files.

##Requirements

- Python >= 3.8
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- Tensorflow/keras (for neural networks)
- GenAI (installed via Ollama or another package depending on setup)
- DeepSeek deepseek-r1:1.5b(for model deployment)
- Flask or FastAPI (for serving model predictions via APIs)

## DeepSeek Deployment Guide for macOS (Apple Silicon/M1/M2)

A step-by-step guide to deploying DeepSeek large language models on Apple Silicon Macs.

#### 📋 System Requirements
- **Hardware**: Mac with M1/M2/M3 chip (8GB RAM minimum, 16GB+ recommended)
- **OS**: macOS Ventura (13.0+) or newer
- **Storage**: 5GB+ free space (model-dependent)
- **Python**: 3.8+ (recommended: 3.10+)

#### 🚀 Quick Start

#### Install Ollama (Optimized for Apple Silicon)

#### Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

#### Install Ollama with Metal acceleration
brew install ollama

#### Start Ollama service
ollama serve

#### Start Ollama service
ollama run deepseek-r1:1.5b

## Data Source

The dataset used for this project is from **Lending Club** and is publicly available at **OpenIntro**. You can find more details about the dataset on the following [link](https://www.openintro.org/data/index.php?data=loans_full_schema).

### Data Details:
- **Source**: [Lending Club Full Schema - OpenIntro](https://www.openintro.org/data/index.php?data=loans_full_schema)
- **Dataset**: Contains loan-level data, including features such as:
  - Loan amount
  - Interest rate
  - Term
  - Grade
  - Employment length
  - Credit score
  - Delinquency status
  - Borrower’s purpose, and more

## Disclaimer
The author of this project is an employee of Wells Fargo. The views and opinions expressed in this repository, including any code, documentation, or analysis, are those of the author and do not reflect the official policies, positions, or opinions of Wells Fargo or any of its subsidiaries.

This project is for educational and research purposes only and is not intended to represent any official products, services, or strategies of Wells Fargo.

