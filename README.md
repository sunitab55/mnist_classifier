# MNIST Handwritten Digit Recognition with PyTorch and Streamlit

This project implements a **handwritten digit recognition system** using a **Convolutional Neural Network (CNN)** trained on the MNIST dataset. The model is developed in **PyTorch** and deployed as a **Streamlit web app** where users can draw digits and get real-time predictions.

## Motivation
Inspired by handwriting recognition features on smartphones, this project allows users to draw numbers on a web-based canvas, and a deep learning model predicts the digit.

## Features

- **CNN-based Digit Recognition**:
  - A **PyTorch-trained CNN** classifies handwritten digits from the MNIST dataset.
  - The trained model is deployed using **Streamlit**.

- **Interactive Web Interface**:
  - Users can draw digits using a mouse or touchscreen.
  - Predictions are displayed in real-time.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- PyTorch & torchvision
- Streamlit
- OpenCV & NumPy (for image processing)

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/mnist-digit-recognition.git
cd mnist-digit-recognition
```

#### 2. Train the Model (Optional)
Run the Jupyter Notebook (`draft.ipynb`) to train the model:
```bash
jupyter notebook draft.ipynb
```
This will:
- Load the MNIST dataset.
- Train a **CNN** model using PyTorch.
- Save the trained model as `model.pth`.

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the Streamlit App
```bash
streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

## How It Works

1. **CNN Training**:
   - A convolutional neural network (CNN) is trained on MNIST using PyTorch.
   - The trained model is saved as `model.pth`.

2. **Web-Based Prediction**:
   - Users draw digits on a canvas.
   - The image is resized, normalized, and fed to the model.
   - The CNN predicts the digit.

3. **Displaying Predictions**:
   - The predicted number is shown below the canvas.

## To-Do

- [ ] Improve accuracy by training on augmented datasets.
- [ ] Deploy the app online (e.g., Streamlit Sharing, Hugging Face Spaces).
- [ ] Optimize frontend performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
