import streamlit as st
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import requests
import os
from inference import predict_image
from train_model_up import train_model

# Function to display Train UI
# Define a function to train the model with hyperparameters
def train_ui():
    st.sidebar.header("Train UI")
    st.sidebar.subheader("Hyperparameters:")
    image_height = st.sidebar.slider("Image Height", min_value=100, max_value=500, value=224, step=1)
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=64, value=16, step=1)
    num_epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=10, value=5, step=1)
    num_classes = st.sidebar.slider("Number of Classes", min_value=2, max_value=50, value=27, step=1)
    learning_rate = st.sidebar.slider("Learning Rate", min_value=1e-6, max_value=1e-2, value=5e-5, step=1e-6)
    seed = st.sidebar.number_input("Seed - For Easy Reproduction", min_value=1, max_value=100000, value=1, step=1)
    st.write("Number Input Value. 1 to 100000", seed)

# Train Button
    if st.sidebar.button("Train Model"):
        train_progress_message = st.sidebar.empty()  # Placeholder for the progress message

        train_progress_message.text("Training in progress...")

        # send a request to start a training process
        training_data = {
            "image_height": image_height,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "seed": seed,

        }
        # Send a request to start the training process
        # fastapi_url = "localhost:8080/train_model"
        fastapi_url = "http://fastapi:8080/train_model"
        response = requests.post(url=fastapi_url, json=training_data)
        if response.status_code == 200:
            train_progress_message.text("Training Complete!")
            model_path = st.text_input("Enter model name:", "my_model.keras")
            if st.button("Save Model"):
                with open(model_path, "wb") as f:
                    f.write(response.content)
                st.write(f"Model saved as {model_path}")
        else:
            st.error("Error occurred during training!")

# Function to display Visualization UI
def visualization_ui():
    st.sidebar.header("Visualization UI")

    # MLflow Experiment Results
    mlflow_runs = mlflow.search_runs()

    # Display Experiment Selector Dropdown
    selected_experiment = st.sidebar.selectbox("Select an Experiment", mlflow_runs['run_id'].tolist())

    # Retrieve selected experiment data
    selected_run_data = mlflow.get_run(selected_experiment).data

    # Accessing metrics
    metrics_dict = selected_run_data.metrics
    metrics = {key: value for key, value in metrics_dict.items()}

    # Display Hyperparameters and Metrics for the selected experiment
    st.subheader("Selected Experiment Data:")
    st.write(f"- Experiment ID: {selected_experiment}")

    # Display metrics
    for key, value in metrics.items():
        st.write(f"- {key}: {value}")

    # Display Training and Validation Losses
    num_epochs = int(metrics.get("num_epochs", 5))
    train_losses = [metrics.get(f"train_loss_epoch_{epoch + 1}", None) for epoch in range(num_epochs)]
    val_losses = [metrics.get(f"validation_loss_epoch_{epoch + 1}", None) for epoch in range(num_epochs)]

    # Display Training and Validation Accuracies
    train_accuracies = [metrics.get(f"train_accuracy_epoch_{epoch + 1}", None) for epoch in range(num_epochs)]
    val_accuracies = [metrics.get(f"validation_accuracy_epoch_{epoch + 1}", None) for epoch in range(num_epochs)]

    # Create dataframe for easier manipulation
    data = {
        "Epoch": range(1, num_epochs + 1),
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "Train Accuracy": train_accuracies,
        "Validation Accuracy": val_accuracies
    }
    df = pd.DataFrame(data)

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df["Epoch"], df["Train Loss"], label='Training Loss')
    plt.plot(df["Epoch"], df["Validation Loss"], label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting the accuracies
    plt.subplot(1, 2, 2)
    plt.plot(df["Epoch"], df["Train Accuracy"], label='Training Accuracy')
    plt.plot(df["Epoch"], df["Validation Accuracy"], label='Validation Accuracy')
    plt.title('Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    st.pyplot(plt)

#Function to display inference UI
def inference_ui():
    st.title("Inference")

    #Scan the 'model/' folder for model files
    model_files = os.listdir("models/")
    model_files = [file for file in model_files if file.endswith('.h5')]

    #Display Model Selector Dropdown
    model_name = st.selectbox("Select Model", model_files)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        if st.button("Predict"):
            predicted_alphabet = predict_image(os.path.join("models", model_name), uploaded_file)
            st.write(f"Predicted Alphabet: {predicted_alphabet}")

# Main code
st.sidebar.title("Navigation")
tabs = ["Train UI", "Visualization UI", "Inference UI"]
selected_tab = st.sidebar.radio("", tabs)

if selected_tab == "Train UI":
    train_ui()
elif selected_tab == "Visualization UI":
    visualization_ui()
elif selected_tab == "Inference UI":
    inference_ui()