import os
import joblib
from data_preprocessing import load_and_preprocess_data
from model_training import train_model
from gui import launch_gui

# Define the dataset path
dataset_path = '/Users/virinchisai/Downloads/PROJECTS/Prodigy Infotech/PRODIGY_ML_01/house-prices-advanced-regression-techniques/train.csv'

# Ensure the model is trained and exists
model_path = 'house_price_model.pkl'

if not os.path.exists(model_path):
    print("Training the model...")
    train_model(dataset_path)

# Load the trained model
model = joblib.load(model_path)

# Launch the GUI
launch_gui(model)
