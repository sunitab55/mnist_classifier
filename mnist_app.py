import streamlit as st
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model (assuming you've saved it as 'mnist_model_scripted.pth')
model = torch.jit.load('mnist_model_scripted.pth')
model.eval()

# Move model to GPU if available
model = model.to(device)

# Set up the Streamlit interface
st.title("MNIST Digit Classifier")

# Create a canvas component for drawing
canvas_result = st_canvas(
    fill_color="white",  # White background
    stroke_width=10,     # Stroke width for drawing
    stroke_color="black",  # Black ink for drawing
    background_color="white",
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Transform for image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor()  # Convert to tensor
    #transforms.Normalize((0.5,), (0.5,))  # Normalize to match the MNIST model's input
])

if canvas_result.image_data is not None:
    # Get the numpy array of the canvas
    input_arr = np.array(canvas_result.image_data)

    # invert the image to match the MNIST
    input_arr = np.invert(input_arr.astype('uint8'))

    threshold = 128
    input_arr[input_arr < threshold] = 0
    input_arr[input_arr >= threshold] = 255

    # Convert the image from the canvas to a PIL image (grayscale)
    image = Image.fromarray(input_arr[:, :, 0].astype('uint8'))  # Get only the first channel

    # Display the image for debugging
    st.image(image, caption="Your Drawing", width=280)

    # Preprocess the image: apply transformations and move to device
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():
        output = model(img_tensor)
        # st.write(f"Model Output (Raw): {output}")  # Display raw model output for debugging

        prediction = output.argmax(dim=1).item()

    # Display the prediction
    st.write(f"Predicted Digit: {prediction}")
