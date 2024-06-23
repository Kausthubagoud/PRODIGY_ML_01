from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import data_preprocessing

def train_model(dataset_path):
    # Load and preprocess data
    features, target = data_preprocessing.load_and_preprocess_data(dataset_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, 'house_price_model.pkl')

    print("Model trained and saved as 'house_price_model.pkl'")

if __name__ == "__main__":
    train_model('train.csv')
