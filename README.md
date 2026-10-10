# Automated Ocular Disease Detection in Retinal Images Using Transfer Learning

## Overview

This project presents a deep learning-based system for classifying retinal fundus images into four categories: **Cataract, Glaucoma, Diabetic Retinopathy, and Normal**. It uses transfer learning with pretrained convolutional neural networks (CNNs) to develop an accurate and computationally efficient image classification model.

Five pretrained architectures were evaluated, and MobileNetV3-Large was selected as the final model based on its classification performance and computational efficiency. The trained model was deployed as an interactive web application using Gradio and Hugging Face Spaces.

## Key Features

- Multi-class classification of retinal fundus images into four categories.
- Comparative evaluation of five pretrained deep learning architectures.
- Transfer learning with a two-stage training strategy: feature extraction and fine-tuning.
- Data augmentation to improve generalization and reduce overfitting.
- Interactive image classification through a Gradio web interface.
- Publicly accessible deployment through Hugging Face Spaces.

## Tech Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Model Architectures:** MobileNetV3-Large, EfficientNetB0, EfficientNetB3, ResNet50, DenseNet201
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Deployment:** Gradio, Hugging Face Spaces
- **Development Environment:** Kaggle Notebooks with NVIDIA Tesla T4 GPUs

## Dataset

The project uses the publicly available Eye Diseases Classification dataset from Kaggle.

**Dataset:** [Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

The dataset contains **4,217 retinal fundus images** distributed across four classes.

| Class | Number of Images |
|---|---:|
| Cataract | 1,038 |
| Diabetic Retinopathy | 1,098 |
| Glaucoma | 1,007 |
| Normal | 1,074 |
| **Total** | **4,217** |

An 80:20 training-validation split was used, with a fixed random seed of 42.

## Methodology

### 1. Data Preprocessing and Augmentation

- Loaded retinal images using TensorFlow's `image_dataset_from_directory`.
- Resized images to 224 × 224 pixels.
- Applied random horizontal flipping, rotations up to ±8°, zoom augmentation, and contrast variations.
- Used the preprocessing expected by the pretrained MobileNetV3 architecture.

### 2. Comparative Model Evaluation

Five pretrained CNN architectures were implemented and evaluated to identify an effective balance between classification performance and computational cost.

- ResNet50
- DenseNet201
- EfficientNetB0
- EfficientNetB3
- MobileNetV3-Large

### 3. Transfer Learning

The final MobileNetV3-Large model was trained in two stages:

- **Feature extraction:** Froze the pretrained backbone and trained the classification head.
- **Fine-tuning:** Unfroze the backbone to adapt the pretrained features to retinal disease classification.

The training pipeline used the Adam optimizer, categorical cross-entropy with label smoothing, and callbacks including ModelCheckpoint, ReduceLROnPlateau, and EarlyStopping.

## Results

The final model comparison reported the following performance.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| ResNet50 | 94% | 95% | 94% | 94% |
| DenseNet201 | 94% | 94% | 94% | 94% |
| EfficientNetB0 | 95% | 95% | 95% | 95% |
| EfficientNetB3 | 95% | 95% | 95% | 95% |
| **MobileNetV3-Large** | **95%** | **95%** | **95%** | **95%** |

MobileNetV3-Large was selected as the final architecture because it achieved competitive classification performance with approximately **9.68 million parameters**, the lowest parameter count among the five evaluated models.

## Deployment

The trained model was exported as `mobilenet_final.keras` and integrated into a Gradio application hosted on Hugging Face Spaces.

The application allows users to:

1. Upload a retinal fundus image.
2. Run the image through the trained model.
3. View the predicted disease category and confidence score.

**Live Demo:** [Try the application on Hugging Face](https://huggingface.co/spaces/penguin18/Retina-disease-classifier)

## Repository Structure

```text
retinal-disease-classification/
├── README.md
├── notebooks/
│   └── retinal_disease_classification.ipynb
├── app.py
├── requirements.txt
└── .gitignore
```

*This is a suggested structure. Adjust the filenames to match the files you upload.*

## Future Improvements

- Validate the model on independent retinal datasets.
- Expand classification to additional ophthalmic conditions.
- Integrate explainability techniques such as Grad-CAM.
- Improve robustness across different retinal imaging conditions.
- Optimize inference for resource-constrained devices.

## Limitations and Disclaimer

This project is intended for educational and research purposes and supports preliminary screening research. It is **not a substitute for professional medical diagnosis**. Predictions may be incorrect, and clinical decisions should be made by qualified ophthalmologists.

## Author

**Riya Mukherjee**  
M.Sc. Computer Science  
National Institute of Technology Tiruchirappalli

- GitHub: [18riyamukherjee03](https://github.com/18riyamukherjee03)
- Hugging Face: [Retina Disease Classifier](https://huggingface.co/spaces/penguin18/Retina-disease-classifier)
