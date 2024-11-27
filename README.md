# Mammo
# ResNet101 Image Classification with Grad-CAM Explanations

This project implements multiple ResNet101 variants for image classification and provides Grad-CAM visual explanations for model interpretability.

---

## Table of Contents
- [Dataset](#dataset)
- [Training](#training)
- [Testing](#testing)
- [Notes](#notes)
- [Files](#files)
- [License](#license)

---

## Dataset

Download the dataset from the provided link. You only need the `images` folder for this project.

---

## Training

### Prerequisites
- Ensure Python 3.8+ is installed.
- Install the required dependencies using:
  ```sh
  pip install -r requirements.txt

## Commands
Assuming you're using a single GPU (GPU 0), you can train the models using the following commands:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py -m mt_prj_resnet101
```

```
CUDA_VISIBLE_DEVICES=0 python3 train.py -m sg_prj_resnet101
```

```
CUDA_VISIBLE_DEVICES=0 python3 train.py -m resnet101
```

<<<<<<< HEAD
## 🧪 Testing
```
python3 test.py
```

## 📝 Notes 
If resuming training for a model previously trained on multiple GPUs, modify utils/auto_load_resume:
```
name = k[:]
```

```
name = k[7:]
```

Model naming conventions:

sg: Single-level projector.
mt: Multi-level projector.

## 📁 Files
train.py: Script for training the models.
test.py: Script for testing the models.
gradcam.py: Script for generating Grad-CAM visual explanations.
config.py: Configuration file for the project.
requirements.txt: List of required packages.
train.csv: Training dataset metadata.
test.csv: Testing dataset metadata.

## 🛠️ Installation
```
pip install -r requirements.txt
```

## 📜 License
This project is licensed under the MIT License.

## 🎯 Example Usage
To train a ResNet101 model with a single-level projector:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py -m sg_prj_resnet101
```
To test the trained model:
=======
## Testing
>>>>>>> 78033ed (up)
```
python3 test.py
```