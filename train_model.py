# train_model.py
# This script trains a Decision Tree Classifier and saves the trained model
# and the list of unique genres using joblib.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib # Library for saving and loading Python objects efficiently

print("Starting model training script...")

# --- 1. Load Data ---
# Load the music dataset from a CSV file.
# Ensure 'music.csv' is in the same directory as this script.
try:
    music_data = pd.read_csv('music.csv')
    print("Successfully loaded 'music.csv'.")
except FileNotFoundError:
    print("Error: 'music.csv' not found. Please ensure the file is in the same directory.")
    exit() # Exit the script if the data file is missing

# --- 2. Prepare Data for Model Training ---
# X (features): Input variables that the model will use to make predictions.
# Here, 'age' and 'gender' are our features. We drop the 'genre' column.
X = music_data.drop(columns=['genre'])
print("Features (X) created: 'age', 'gender'.")

# y (target): The variable we want to predict.
# Here, 'genre' is our target variable.
y = music_data['genre']
print("Target (y) created: 'genre'.")

# Split the data into training and testing sets.
# X_train, y_train: Data used to train the model.
# X_test, y_test: Data used to evaluate the model's performance on unseen data.
# test_size=0.2 means 20% of the data will be used for testing.
# random_state=42 ensures reproducibility, so results are consistent every time.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")

# --- 3. Train the Model ---
# Initialize the Decision Tree Classifier model.
model = DecisionTreeClassifier()
print("Decision Tree Classifier model initialized.")

# Train the model using the training data.
# The model learns patterns from X_train to predict y_train.
model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Evaluate the Model (Optional but good practice) ---
# Make predictions on the test set to see how well the trained model performs.
predictions = model.predict(X_test)

# Calculate the accuracy score.
# This measures the proportion of correct predictions on the test set.
score = accuracy_score(y_test, predictions)
print(f"Model Accuracy on Test Set: {score:.2f}")

# --- 5. Save the Trained Model and Genres List ---
# Save the trained model to a file using joblib.
# This allows the Streamlit app to load the model without retraining.
joblib.dump(model, 'trained_model.pkl')
print("Trained model saved as 'trained_model.pkl'.")

# Get the unique music genres from the dataset.
# This list is useful for displaying options or validating predictions in the app.
genres = music_data['genre'].unique().tolist()
joblib.dump(genres, 'genres.pkl')
print("Unique genres list saved as 'genres.pkl'.")

print("Model training and saving process finished.")
