# MNIST Handwritten Digit Recognition App with ONNX and React

This project is inspired by a feature commonly found on smartphones where you can draw letters or digits on the keyboard, and it automatically recognizes what you're trying to write. I wanted to replicate this functionality by creating a web app where users can draw digits, and a machine learning model will guess what number they wrote in real-time. The app is built using **Next.js** (React), **TensorFlow.js** for image processing, and **ONNX Runtime** for running the digit recognition model. The actual model trained is in the draft.ipynb using PyTorch.

### Motivation
Inspired by the drawing features on smartphones that recognize letters and digits, this app replicates that experience by allowing users to draw numbers directly on a website, which are then processed by a machine learning model trained on the MNIST dataset.

## Features

- **Handwritten Digit Recognition**:
  - A pre-trained ONNX model (loaded via `onnxruntime-web`) recognizes digits drawn by users in real-time.
  
- **Canvas Drawing Interface**:
  - Users can draw a digit using their mouse or finger on a responsive canvas in the browser.
  
- **Real-time Predictions**:
  - After drawing a digit, users can click "Predict" to get an instant prediction of the digit. The app will display the result below the canvas.

- **Clear Button**:
  - A button to reset the canvas and allow users to redraw as many times as they like.

## Getting Started

To get the app running locally, follow these steps:

### Prerequisites

Make sure you have the following installed:
  
- Node.js
- Python (for model training, if needed)
- TensorFlow.js (for processing image data)
- ONNX Runtime for Web (for model inference in the browser)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/mnist-digit-recognition-app.git
    cd mnist-digit-recognition-app
    ```

2. **Install frontend dependencies**:
    ```bash
    cd frontend
    npm install
    ```

3. **Model Setup**:
    - Download or place your trained ONNX model (`model.onnx`) in the correct directory.

4. **Run the app**:
    ```bash
    npm run dev
    ```

5. Visit `http://localhost:3000` in your browser to interact with the app.

## How It Works

1. **Canvas Drawing**:
   - The app allows users to draw digits on a 280x280 canvas using mouse or touch events.

2. **Image Processing**:
   - The drawing is resized to 28x28 pixels, converted to grayscale, and normalized using TensorFlow.js.

3. **Model Prediction**:
   - The processed image is sent to an ONNX model running in the browser, and the model predicts which digit was drawn.
   
4. **Displaying Results**:
   - The predicted digit is shown below the canvas.

## To-Do

- [ ] Improve accuracy with additional training.
- [ ] Add cloud deployment (AWS/GCP/Azure).
- [ ] Improve mobile responsiveness.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
