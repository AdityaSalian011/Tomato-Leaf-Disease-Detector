import streamlit as st
import torchvision.models as models
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import os
import gdown

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# STREAMLIT BASICS (Read this before going further!)
# ============================================================
# Streamlit works top-to-bottom, just like a regular Python script.
# Every time the user interacts (uploads a file, clicks a button),
# the entire script reruns from top to bottom automatically.
#
# Key Streamlit functions you'll see below:
#   st.title()       → Big heading on the page
#   st.write()       → Write any text or data on the page
#   st.image()       → Display an image
#   st.success()     → Green success message box
#   st.error()       → Red error message box
#   st.warning()     → Yellow warning message box
#   st.spinner()     → Show a loading animation while something runs
#   st.file_uploader()→ Upload file widget
#   st.columns()     → Split page into side-by-side columns
#   st.sidebar       → Left sidebar panel
# ============================================================


# ============================================================
# STEP 1 — PAGE CONFIGURATION
# This must be the FIRST streamlit command in your script.
# ============================================================
st.set_page_config(
    page_title='Tomato Leaf Disease Detector',
    page_icon='🍅',
    layout='centered'   # "centered" or "wide"
)

# ============================================================
# STEP 2 — DEFINE IMAGE TRANSFORMING CONFIG
# ============================================================
def apply_transformer(image):
    """Apply Image Transforming settings"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(image).unsqueeze(0).to(device)

# ============================================================
# STEP 3 — LOAD THE TRAINED MODEL
# ============================================================
# @st.cache_resource is a decorator that makes sure the model is
# loaded ONLY ONCE when the app starts, and then reused for every
# prediction. Without this, the model would reload on every interaction,
# making the app very slow.
# ============================================================
@st.cache_resource
def load_model():
    """Loading our pretrained MobileNet model"""

    # -------------------------------------------------------
    # Download model from Google Drive if not already present
    # Replace the ID below with your actual file ID
    # -------------------------------------------------------
    model_path = "mobilenet_tomato_leaf_detector.pt"

    if not os.path.exists(model_path):
        with st.spinner('Downloading model... (first time only)'):
            file_id = '1845vELPMxnYkqgweZZSmi5DI3hY5rpVj'
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

    mobilenet = models.mobilenet_v2(weights=None)

    mobilenet.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 3)
    )

    mobilenet.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    mobilenet = mobilenet.to(device)
    mobilenet.eval()

    return mobilenet

# ============================================================
# STEP 4 — PREDICTION FUNCTION
# Runs the preprocessed image through the model and returns
# the predicted class and confidence score.
# ============================================================
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Recommended action for each detected disease
RECOMMENDATIONS = {
    "Early Blight": (
        "⚠️ Early Blight Detected!\n\n"
        "**Recommended Actions:**\n"
        "- Remove and destroy infected leaves immediately.\n"
        "- Apply copper-based fungicide spray.\n"
        "- Avoid overhead watering; water at the base of the plant.\n"
        "- Ensure proper spacing between plants for air circulation."
    ),
    "Late Blight": (
        "🚨 Late Blight Detected! (Serious - Act Quickly)\n\n"
        "**Recommended Actions:**\n"
        "- Remove and burn all infected plant parts immediately.\n"
        "- Apply Mancozeb or Chlorothalonil fungicide.\n"
        "- Do NOT compost infected material.\n"
        "- Consult your local agricultural officer if spreading rapidly."
    ),
    "Healthy": (
        "✅ Leaf is Healthy!\n\n"
        "**Tips to Keep it Healthy:**\n"
        "- Continue regular watering and fertilization.\n"
        "- Monitor weekly for early signs of disease.\n"
        "- Maintain proper plant spacing.\n"
        "- Use neem oil spray as a natural preventive measure."
    )
}
def predict(model, image):
    """
    Pass the preprocessed image to the model and return
    the predicted class name and confidence percentage.
    """
    
    output = model(image)             # Get raw output probabilities for each class
    _, predicted = torch.max(output, 1)    # Find the index of the highest probability
    probs = torch.softmax(output, dim=1)    # Get probability of each class
    confidence = probs[0][predicted.item()].item()   # Get the confidence score for that class
    class_name = CLASS_NAMES[predicted.item()]      # Map index → class name
    return class_name, confidence

# ============================================================
# STEP 5 — SIDEBAR (Left panel with app info)
# st.sidebar works just like st.write() but renders on the left side
# ============================================================
with st.sidebar:
    st.title("ℹ️ About This App")
    st.write(
        "This app uses a Pretrained MobileNet Model "
        "fine-tuned on the PlantVillage dataset to detect diseases "
        "in tomato plant leaves."
    )
    st.divider()  # Draws a horizontal line
    st.write("**Detectable Conditions:**")
    st.write("🟡 Early Blight")
    st.write("🔴 Late Blight")
    st.write("🟢 Healthy Leaf")
    st.divider()
    st.write("**How to use:**")
    st.write("1. Upload a clear photo of a tomato leaf.")
    st.write("2. Wait for the prediction.")
    st.write("3. Follow the recommended action.")
    st.divider()
    st.caption("S.Y.C.S | Community Engagement Project")

# ============================================================
# STEP 6 — MAIN PAGE CONTENT
# ============================================================
# st.title() → Displays a large H1 heading
st.title("🍅 Tomato Leaf Disease Detector")

# st.write() → Renders text (supports Markdown formatting)
st.write(
    "Upload a photo of a tomato leaf and the AI model will detect "
    "whether it is **Healthy**, has **Early Blight**, or **Late Blight**."
)

# st.divider() → Just a visual horizontal line separator
st.divider()

# ============================================================
# STEP 7 — FILE UPLOADER
# st.file_uploader() creates an upload widget on the page.
# 'type' restricts the accepted file formats.
# Returns None if no file is uploaded yet.
# ============================================================
uploaded_file = st.file_uploader(
    label='📤 Upload a Tomato Leaf Image',
    type=['jpg', 'jpeg', 'png']
)

# ============================================================
# STEP 8 — PREDICTION LOGIC
# This block only runs when the user has uploaded a file.
# ============================================================
if uploaded_file:

    # Open the uploaded image using PIL
    image = Image.open(uploaded_file).convert('RGB')

    # st.columns() → Splits the page into side-by-side columns
    # col1 will show the image, col2 will show the result
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('📷 Uploaded Image')
        # st.image() → Displays an image. use_container_width stretches it to column width.
        st.image(image, use_container_width=True)

    with col2:
        st.subheader('🔍 Prediction Result')

        # st.spinner() → Shows a loading animation while the indented code runs
        with st.spinner('Analyzing leaf... please wait'):
            try:
                model = load_model()    # Load model (cached after first load)

                processed = apply_transformer(image)    # Preprocess the uploaded image
                class_name, confidence = predict(model, processed)  # Get prediction

                # Display result based on predicted class
                if class_name == 'Healthy':
                    # st.success() → Green box
                    st.success(f'**{class_name}**')
                elif class_name == "Early Blight":
                    # st.warning() → Yellow/orange box
                    st.warning(f"**{class_name}**")
                elif class_name == 'Late Blight':
                    # st.error() → Red box
                    st.error(f"**{class_name}**")

                # Display confidence score as a percentage
                st.metric(
                    label='Confidence Score',
                    value=f'{confidence*100:.2f}%'
                )

            except Exception as e:
                st.error(f"Model not loaded yet. Error: {e}")
                st.info("💡 Make sure 'mobilenet_tomato_leaf_detector.pt' is available or check your Google Drive file ID.")

    # ============================================================
    # STEP 9 — RECOMMENDATIONS SECTION
    # Shown below the image and result columns
    # ============================================================
    st.divider()
    st.subheader('📋 Recommended Action for Farmer')

    try:
        recommendations = RECOMMENDATIONS[class_name]
        # st.write() renders Markdown, so **text** becomes bold
        st.write(recommendations)
    except:
        pass

else:
    # This shows when no image is uploaded yet (default state)
    st.info("👆 Please upload a tomato leaf image above to get started.")
