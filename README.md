# DINOv2 Fine-Tuning and Production Benchmarking Pipeline

## Overview
This project is dedicated to implementing fine-tuning for the DINOv2 model and setting up a production benchmarking pipeline. The aim is to enhance the model's performance on specialized tasks by fine-tuning and to ensure robust, scalable deployment through comprehensive benchmarking.

## Model Details
The DINOv2 model, based on the Vision Transformer (ViT) architecture, is enhanced in this project to better adapt to specific domains or tasks through fine-tuning on targeted datasets.

## Benchmarking Strategy
This benchmarking pipeline evaluates the model on multiple fronts: performance metrics (accuracy, precision, recall), latency, and throughput under various system loads.


# Implementing the project

## Project Structure
```
├── finetune.py                   # Script for fine-tuning DINOv2 on new datasets
├── load_data.py                  # Data loading utilities
├── model.py                      # DINOv2 model definition and initial setup
├── production_benchmarking.py    # Benchmarking model performance in production
└── sarl_env_xformers.yml         # Project environment config
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.8 or higher
- torchvision
- timm (for PyTorch image models)

### Data Preparation
Use `load_data.py` to prepare the data for fine-tuning:
```bash
python load_data.py --data_path './data/'
```

### Fine-Tuning
To start fine-tuning the DINOv2 model with your dataset, run:
```bash
python finetune.py --data_path './data/' --output_path './models/'
```

### Benchmarking
Evaluate the performance of the fine-tuned model in a production-like environment using:
```bash
python production_benchmarking.py --model_path './models/model_final.pth'
```

## Contact
For any queries or further assistance, please contact Jordan Nieusma (jordan.m.nieusma@vanderbilt.edu)
