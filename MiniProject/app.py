import os
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# --- 1. App & Path Configuration ---
app = Flask(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "Deepfake_image_detection_project" # The folder on your desktop

# --- 2. Image Transformation ---
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. Model Architecture Definitions ---

# Architecture for:
# - min-dalle-simple-cnn_best_model.pth
# - stable-diffusion-simple-cnn_best_model.pth
# - openjourney-simple-cnn_best_model.pth
class SimpleCNN_v1(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN_v1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Architecture for:
# - simple_cnn_v2_best_model.pth (formerly simple_cnn_best_model(1).pth)
class SimpleCNN_v2_Dropout(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN_v2_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # The key difference
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x) # Applying dropout
        x = self.fc2(x)
        return x

# --- REMOVED PHOTOSHOP (RESNET) MODEL ---
# The ResNet18Model function was here

# --- 4. Model Loading ---

def load_pytorch_model(model_class_or_fn, model_filename):
    """A helper function to load a model and set it to eval mode."""
    try:
        model = model_class_or_fn(num_classes=2)
        model_file = os.path.join(MODEL_PATH, model_filename)
        
        if not os.path.exists(model_file):
            print(f"❌ ERROR: Model file not found at {model_file}")
            return None
            
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set to evaluation mode
        print(f"✅ Successfully loaded {model_filename}")
        return model
    except Exception as e:
        print(f"❌ ERROR loading {model_filename}: {e}")
        return None

print("--- Loading all models... ---")
MODELS = {
    # These names ("StyleGAN", etc.) are what will show up in the UI.
    "StyleGAN": load_pytorch_model(SimpleCNN_v2_Dropout, "simple_cnn_v2_best_model.pth"),
    # --- REMOVED PHOTOSHOP MODEL FROM THIS LIST ---
    "Stable Diffusion": load_pytorch_model(SimpleCNN_v1, "stable-diffusion-simple-cnn_best_model.pth"),
    "Openjourney": load_pytorch_model(SimpleCNN_v1, "openjourney-simple-cnn_best_model.pth"),
    "Min-DALL-E": load_pytorch_model(SimpleCNN_v1, "min-dalle-simple-cnn_best_model.pth")
}
print("--- All models loaded. ---")

# --- 5. Prediction Function (Average of Top 2 Logic) ---

def predict_image(image_pil):
    """Runs the uploaded image through all models."""
    image_tensor = inference_transform(image_pil).unsqueeze(0).to(DEVICE)
    
    model_predictions = {}
    all_scores = [] # To store all individual scores
    
    with torch.no_grad():
        for model_name, model in MODELS.items():
            if model is None:
                model_predictions[model_name] = 0 # Model failed to load
                all_scores.append(0.0) # Add 0 to scores list
                continue
                
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # 0 = FAKE, 1 = REAL. We want the "deepfake" probability (index 0).
            fake_prob = probabilities[0][0].item() * 100
            
            model_predictions[model_name] = round(fake_prob)
            all_scores.append(fake_prob) # Add the score to the list

    # --- This logic automatically adapts ---
    if all_scores:
        all_scores.sort(reverse=True) 
        top_two_scores = all_scores[:2] 
        overall_score = round(sum(top_two_scores) / len(top_two_scores))
    else:
        overall_score = 0
    # --- END OF LOGIC ---
        
    # Format the final dictionary
    results = {
        "overall_deepfake_score": overall_score,
        "generative_ai_models": model_predictions
    }
    
    return results

# --- 6. Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handles the image upload and returns detection results."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected image file"}), 400

    try:
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = predict_image(image_pil)
        return jsonify(results)

    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

# --- 7. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)