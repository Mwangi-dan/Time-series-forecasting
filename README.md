# Time Series Forecasting Model for PM2.5 Prediction

## Overview

This project focuses on building a **Time Series Forecasting Model** to predict **PM2.5 air pollution levels** using **Recurrent Neural Networks (RNNs)**, specifically **LSTMs** and **GRUs**. Accurate forecasting of pollutant levels can help guide public health initiatives and policy decisions.

## Features

- **Dataset Exploration**: Statistical analysis and visualizations (line plots, histograms, and box plots) to identify trends and anomalies.
- **Preprocessing & Feature Engineering**: Data normalization, missing value handling, and temporal feature extraction.
- **Deep Learning Models**: Implementation of LSTMs and GRUs with different architectures.
- **Experimentation**: 24 model variations tested with different hyperparameters.
- **Performance Evaluation**: Comparison using RMSE and MAE metrics.
- **Optimization Techniques**: Batch normalization, dropout, and gradient clipping to improve stability.

## Dataset

The dataset consists of **PM2.5 concentration levels** and other **meteorological variables** over time. It includes:

- Time-based variables (hour, seasonality)
- Air quality index values
- Missing data handled using forward-filling and mean imputation

## Model Architecture

A **Bidirectional LSTM-GRU Hybrid Network** was found to be the best-performing model. The experiments involved:

- Varying the **number of layers, hidden units, dropout rates, and batch sizes**.
- Testing different **optimizers** (Adam, RMSprop, SGD).
- Evaluating different **loss functions** (MSE, BCE, Huber loss).

## Key Results

- The best-performing model achieved **significant RMSE reduction** compared to the baseline.
- **Adam optimizer** provided the best convergence speed.
- **Batch normalization** and **dropout** helped reduce overfitting.
- **More layers improved feature extraction** but introduced vanishing gradient issues, handled using GRUs and batch normalization.

## Challenges & Solutions

- **Vanishing gradients**: Mitigated using **GRU layers** and **batch normalization**.
- **Exploding gradients**: Controlled through **gradient clipping**.
- **Data imbalance**: Addressed using **resampling and SMOTE methods**.

## Future Improvements

- Implementing **attention mechanisms** to enhance forecasting accuracy.
- Exploring **Transformer-based models** for improved sequence learning.
- Incorporating additional meteorological data for better predictions.


