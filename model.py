import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

def train_and_save_model(file_path):
    """
    Loads data, trains a Random Forest model, and saves it.
    This version includes synthetic image-derived features for a complete model.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return

    # Add synthetic image-derived features to the DataFrame
    # These columns simulate the output of our image_analysis.py script
    df['crack_severity'] = np.random.uniform(0.0, 1.0, len(df))
    df['erosion_index'] = np.random.uniform(0.0, 1.0, len(df))
    
    # Select all relevant features for the model, including the new ones
    features = [
        'Elevation', 'Slope', 'Rainfall_mm', 'Snow_mm', 
        'Temperature_C', 'Wind_speed_kmh', 'Fracture_Density',
        'crack_severity', 'erosion_index'
    ]
    target = 'Rockfall_Event'

    if not all(col in df.columns for col in features + [target]):
        print("Error: Required columns are missing from the CSV file.")
        print(f"Expected: {features + [target]}")
        print(f"Found: {list(df.columns)}")
        return

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training the Random Forest model with combined features...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Performance on Test Set:")
    print(f"Accuracy: {accuracy:.2f}")

    joblib.dump(model, 'rockfall_model.joblib')
    print("\nModel saved as 'rockfall_model.joblib'")
    joblib.dump(features, 'model_features.joblib')
    print("Feature list saved as 'model_features.joblib'")

if __name__ == "__main__":
    train_and_save_model('balanced_synthetic_rockfall_data.csv')