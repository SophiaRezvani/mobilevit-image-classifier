# --- SECTION 1: INSTALLS & IMPORTS ---
# Run !pip install -q gradio transformers in a separate cell if not already installed
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from PIL import Image
import requests
import torch
import torch.nn.functional as F
import gradio as gr

# --- SECTION 2: MODEL LOADING ---
# We load the model once at the top to save memory and time
model_name = "apple/mobilevit-small"
processor = MobileViTImageProcessor.from_pretrained(model_name)
model = MobileViTForImageClassification.from_pretrained(model_name)

# --- SECTION 3: THE CORE LOGIC ---
def predict_with_confidence(image_input):
    """
    Takes a URL or a local file path, processes it through MobileViT,
    and returns a human-readable label with a confidence score.
    """
    try:
        # 1. Handle Input (URL vs Local)
        if str(image_input).startswith('http'):
            image = Image.open(requests.get(image_input, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")
        
        # 2. Pre-process the image
        inputs = processor(images=image, return_tensors="pt")
        
        # 3. Run Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 4. Calculate Results
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        predicted_class_idx = logits.argmax(-1).item()
        label = model.config.id2label[predicted_class_idx]
        confidence = probs[0][predicted_class_idx].item()
        
        return f"Prediction: {label} | Confidence: {confidence:.2%}"
    
    except Exception as e:
        return f"Error: {str(e)}"

# --- SECTION 4: TESTING (CONSOLE) ---
print("--- RUNNING TESTS ---")
url_test = "http://images.cocodataset.org/val2017/000000039769.jpg"
print(f"URL Test: {predict_with_confidence(url_test)}")

# To test your local image, uncomment the line below and paste your path:
# my_local_path = "/kaggle/input/test-image/my_picture.png"
# print(f"Local Test: {predict_with_confidence(my_local_path)}")

# --- SECTION 5: INTERACTIVE INTERFACE ---
# This creates a web UI inside your notebook
demo = gr.Interface(
    fn=predict_with_confidence, 
    inputs=gr.Image(type="filepath", label="Upload Image or Drag & Drop"),
    outputs=gr.Textbox(label="Model Output"),
    title="MobileViT Edge-AI Classifier",
    description="A lightweight Computer Vision model designed for mobile efficiency."
)

# Set share=True to get a public link you can send to recruiters
demo.launch(share=True)
