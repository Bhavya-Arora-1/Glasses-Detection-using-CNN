# Glasses-Detection-using-CNN


## Overview
This project implements a **Convolutional Neural Network (CNN)** to classify images of individuals as either wearing glasses or not. It is designed to explore the applications of deep learning in image classification tasks, specifically focusing on facial attributes.

## Features
- **Image Preprocessing**: Includes resizing, normalization, and augmentation to enhance the dataset.
- **Model Architecture**: Employs a CNN with multiple convolutional, pooling, and dense layers for feature extraction and classification.
- **Performance Metrics**: Evaluates the model using accuracy, precision, recall, and F1-score.
- **Visualization**: Displays training performance and prediction results using plots.

## Dataset
- The dataset contains labeled images of individuals, divided into two categories:
  - **With Glasses**: Images of individuals wearing glasses.
  - **Without Glasses**: Images of individuals not wearing glasses.
- Preprocessed to ensure uniformity in size and format, and augmented to improve model generalization.
- The dataset is stored in the `data/` directory and organized as:
  ```
  data/
    train/
      with_glasses/
      without_glasses/
    test/
      with_glasses/
      without_glasses/
  ```

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow/Keras
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, OpenCV

## Prerequisites
- Python 3.8+
- TensorFlow 2.x

## Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/Bhavya-Arora-1/glasses-detection.git
   cd glasses-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Place your image dataset in the `data/` directory.
   - Ensure the dataset is divided into `train/` and `test/` subdirectories.

4. Train the model:
   ```bash
   python train.py
   ```

5. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Results
- Achieved high accuracy on the test dataset.
- Visualizations include loss and accuracy curves, as well as sample predictions.

## Contributions
Feel free to fork this repository, submit issues, or open pull requests for improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
