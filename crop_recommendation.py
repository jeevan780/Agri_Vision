import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load the dataset
crop = pd.read_csv("Crop_recommendation.csv")

# Display dataset information
print("Dataset Information:")
crop.info()
print("\nFirst 5 rows:")
print(crop.head())

# Splitting features and target variable
X = crop.drop(columns=["label"])
y = crop["label"]

# Encoding labels
le = LabelEncoder()
y = le.fit_transform(y)  # Converts crop names to numerical labels

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)

# Train the Random Forest model with optimized parameters
rfc = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rfc.fit(X_train, y_train)

# Model Evaluation
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Function to recommend a crop
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    transformed_features = ms.transform(features)  # Use transform, not fit_transform
    prediction = rfc.predict(transformed_features)
    recommended_crop = le.inverse_transform(prediction)[0]  # Convert back to crop name
    print(f"Recommended Crop: {recommended_crop}")
    return recommended_crop

# Example Predictions
#print("\nExample Predictions:")
#recommend_crop(40, 50, 50, 40.0, 20.0, 6.5, 100)
#recommend_crop(100, 90, 100, 50.0, 90.0, 6.8, 202.0)

# Save the trained model and scaler
pickle.dump(rfc, open("model.pkl", "wb"))
pickle.dump(ms, open("scaler.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))
