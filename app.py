import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
from PIL import Image

# ------------------------ Load the Model ------------------------ #

# Define the custom classifier head
class CustomHead(nn.Module):
    def __init__(self, in_features, num_classes=7):  # Adjust num_classes based on dataset
        super(CustomHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model with cached resource
@st.cache_resource
def load_model():
    device = torch.device("cpu")  # Use CPU for Streamlit
    model = models.resnet50(pretrained=False)  # Do NOT load ImageNet weights
    model.fc = CustomHead(model.fc.in_features, num_classes=7)  # Ensure correct number of classes
    model.load_state_dict(torch.load("./resnet50_ham10000_model2.pth", map_location=device))  # Load trained weights
    model.eval()  # Set model to evaluation mode
    return model

model = load_model()

# ------------------------ Load CSV Labels ------------------------ #

df = pd.read_csv("./sample_images.csv")  # Ensure 'image_id' and 'dx' columns exist

# Define class labels (modify based on dataset)
class_labels = {
    0: "bkl",
    1: "nv",
    2: "df",
    3: "mel",
    4: "vasc",
    5: "bcc",
    6: "akiec"
}


# ------------------------ Image Preprocessing ------------------------ #

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# ------------------------ Streamlit UI ------------------------ #

st.title("Skin Lesion Classifier")
st.write("Upload an image to classify the type of skin lesion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure image is RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)

    predicted_index = torch.argmax(output).item()  # Get predicted class index
    predicted_class = class_labels.get(predicted_index, "Unknown")  # Map to class name

    # Fetch actual label from CSV
    image_id = uploaded_file.name.split(".")[0]  # Extract ID from filename
    actual_label = df[df["image_id"] == image_id]["dx"].values
    actual_label = actual_label[0] if len(actual_label) > 0 else "Unknown"

    # Display results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Actual Label:** {actual_label}")
