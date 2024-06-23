import pandas as pd

def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)

    # Fill or drop missing values as necessary
    data = data.dropna(subset=['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice'])

    # Select features and target variable
    features = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
    target = data['SalePrice']
    
    return features, target

if __name__ == "__main__":
    features, target = load_and_preprocess_data('train.csv')
    print(features.head())
    print(target.head())
