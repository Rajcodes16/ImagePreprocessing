# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import gradio as gr
import streamlit as st

# 1. Data Preprocessing and Augmentation
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
data_dir = './data'
dataset = datasets.FakeData(transform=data_transforms)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 2. Model Training
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 3. Model Inference and Metrics Calculation
def predict(image):
    model.eval()
    with torch.no_grad():
        image = data_transforms(image).unsqueeze(0).to(device)
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_probs, top_classes = torch.topk(probs, 3)
        predictions = [(int(cls), float(prob)) for cls, prob in zip(top_classes, top_probs)]
        return predictions

def calculate_metrics():
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    return precision, recall

# 4. Gradio Interface
def gradio_interface(image):
    predictions = predict(image)
    precision, recall = calculate_metrics()
    metrics_table = pd.DataFrame({"Class": [0, 1], "Precision": precision, "Recall": recall})
    result = "\n".join([f"Class: {cls}, Confidence: {conf:.2f}" for cls, conf in predictions])
    return result, metrics_table

gr.Interface(fn=gradio_interface, inputs=gr.Image(), outputs=["text", "dataframe"]).launch()

# 5. Streamlit Interface
def streamlit_interface():
    st.title("Image Classification App")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        predictions = predict(image)
        st.write("## Predictions")
        for cls, conf in predictions:
            st.write(f"Class: {cls}, Confidence: {conf:.2f}")
        
        st.write("## Precision and Recall")
        precision, recall = calculate_metrics()
        metrics_table = pd.DataFrame({"Class": [0, 1], "Precision": precision, "Recall": recall})
        st.dataframe(metrics_table)

if __name__ == "__main__":
    streamlit_interface()

