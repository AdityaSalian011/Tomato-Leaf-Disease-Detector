import torchvision.models as models
import torch
import torch.nn as nn
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def load_model():
    model_path = "mobilenet_tomato_leaf_detector.pt"  
    
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