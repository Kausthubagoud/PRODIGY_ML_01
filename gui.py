import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import pandas as pd
import joblib

def launch_gui(model):
    # Function to predict house price
    def predict_price():
        try:
            sqft = float(entry_sqft.get())
            bedrooms = int(entry_bedrooms.get())
            bathrooms = int(entry_bathrooms.get())
            
            input_data = pd.DataFrame({
                'GrLivArea': [sqft],
                'BedroomAbvGr': [bedrooms],
                'FullBath': [bathrooms]
            })
            
            predicted_price = model.predict(input_data)[0]
            result_label.config(text=f"Predicted Price: ${predicted_price:,.2f}")
        except ValueError:
            result_label.config(text="Please enter valid numbers")

    # Setting up the GUI
    root = tk.Tk()
    root.title("House Price Predictor")

    # Load and resize the image
    image = Image.open("/Users/virinchisai/Downloads/PROJECTS/Prodigy Infotech/PRODIGY_ML_01/house-prices-advanced-regression-techniques/house_logo.png")  # Path to your house logo image
    image = image.resize((100, 100), Image.ANTIALIAS)
    house_logo = ImageTk.PhotoImage(image)

    # Create a style for the GUI
    style = ttk.Style()
    style.configure('TLabel', font=('Arial', 12))
    style.configure('TEntry', font=('Arial', 12))
    style.configure('TButton', font=('Arial', 12))

    # GUI Components
    ttk.Label(root, image=house_logo).grid(row=0, column=0, columnspan=2, padx=10, pady=10)
    ttk.Label(root, text="Square Footage (GrLivArea):").grid(row=1, column=0, padx=10, pady=10)
    entry_sqft = ttk.Entry(root)
    entry_sqft.grid(row=1, column=1, padx=10, pady=10)

    ttk.Label(root, text="Number of Bedrooms (BedroomAbvGr):").grid(row=2, column=0, padx=10, pady=10)
    entry_bedrooms = ttk.Entry(root)
    entry_bedrooms.grid(row=2, column=1, padx=10, pady=10)

    ttk.Label(root, text="Number of Bathrooms (FullBath):").grid(row=3, column=0, padx=10, pady=10)
    entry_bathrooms = ttk.Entry(root)
    entry_bathrooms.grid(row=3, column=1, padx=10, pady=10)

    predict_button = ttk.Button(root, text="Predict Price", command=predict_price)
    predict_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    result_label = ttk.Label(root, text="Predicted Price: $0.00")
    result_label.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

    root.mainloop()

# For testing purposes, create a dummy model
class DummyModel:
    def predict(self, input_data):
        return [100000]  # Dummy prediction

# Launch GUI with a dummy model
dummy_model = DummyModel()
launch_gui(dummy_model)
