# app.py
# This script creates an interactive Streamlit dashboard
# for predicting music genre preferences based on age and gender.

import streamlit as st
import pandas as pd
import joblib # Used for loading the pre-trained machine learning model
import matplotlib.pyplot as plt # For creating static plots
import seaborn as sns # For enhanced data visualizations

# --- 1. Dashboard Configuration and Title ---
# Set general page configurations for the Streamlit app.
st.set_page_config(
    page_title="Music Preference Predictor", # Title that appears in the browser tab
    layout="centered", # Page layout (can be "centered" or "wide")
    initial_sidebar_state="auto" # State of the sidebar (if any)
)

st.title("🎶 Music Preference Predictor Dashboard") # Main title of the application
st.markdown("---") # A horizontal rule for visual separation

# --- 2. Load Data and Pre-trained Model ---
# Use st.cache_data to cache data loading.
# This decorator ensures the data is loaded only once, even if the app reruns,
# improving performance for static data.
@st.cache_data
def load_data():
    """Loads the music dataset from a CSV file."""
    try:
        data = pd.read_csv('music.csv')
        return data
    except FileNotFoundError:
        st.error("Error: 'music.csv' not found. Please ensure it's in the same directory.")
        st.stop() # Stop the app execution if data is missing

# Load the dataset
music_data = load_data()

# Load the trained machine learning model and the list of unique genres.
# These files are created by 'train_model.py'.
try:
    model = joblib.load('trained_model.pkl')
    genres_list = joblib.load('genres.pkl') # List of all possible genres
    st.sidebar.success("Model and data loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'trained_model.pkl' or 'genres.pkl' not found. Please run 'train_model.py' first.")
    st.stop() # Stop the app if model files are missing

# --- 3. Dashboard Introduction ---
st.write(
    """
    Welcome to the Music Preference Predictor! This interactive dashboard uses a **Decision Tree Machine Learning model**
    to predict a user's likely music genre preference based on their **age** and **gender**.

    **How it works:**
    * The model was trained on a small dataset mapping age and gender to music genres.
    * You can input age and gender below to get an instant prediction.
    * Explore the data insights and model performance sections for more details.
    """
)

# --- 4. Display Raw Data Sample (Optional but good for transparency) ---
st.header("📊 Original Data Sample")
st.write("A small sample of the data used to train the model:")
st.dataframe(music_data.head(5)) # Display the first 5 rows of the dataset
st.markdown("---")

# --- 5. Exploratory Data Analysis (EDA) - Visualizations ---
st.header("📈 Data Insights and Distributions")
st.write("Understand the characteristics of the training data through visualizations.")

# Create a temporary 'gender_label' column for better plot readability
# This maps the numerical gender (0, 1) to descriptive labels ('Female', 'Male').
music_data['gender_label'] = music_data['gender'].map({1: 'Male', 0: 'Female'})

# Plot 1: Music Genre Distribution
st.subheader("1. Distribution of Music Genres")
fig1, ax1 = plt.subplots(figsize=(8, 5)) # Create a matplotlib figure and axes
sns.countplot(y='genre', data=music_data, order=music_data['genre'].value_counts().index, palette='viridis', ax=ax1)
ax1.set_title('Count of Each Music Genre in Dataset')
ax1.set_xlabel('Number of Entries')
ax1.set_ylabel('Music Genre')
st.pyplot(fig1) # Display the plot in Streamlit

# Plot 2: Age Distribution
st.subheader("2. Distribution of Ages")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.histplot(music_data['age'], bins=5, kde=True, color='skyblue', ax=ax2)
ax2.set_title('Distribution of Ages in Dataset')
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')
st.pyplot(fig2)

# Plot 3: Genre Preference by Gender
st.subheader("3. Music Genre Preference by Gender")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.countplot(y='genre', hue='gender_label', data=music_data, palette='coolwarm', ax=ax3)
ax3.set_title('Music Genre Preference by Gender in Dataset')
ax3.set_xlabel('Number of Entries')
ax3.set_ylabel('Music Genre')
ax3.legend(title='Gender') # Add a legend for gender
st.pyplot(fig3)

# Remove the temporary 'gender_label' column to keep the original DataFrame clean
music_data.drop(columns=['gender_label'], inplace=True)
st.markdown("---")

# --- 6. Interactive Prediction Interface ---
st.header("🔮 Get Your Personalized Music Genre Prediction!")
st.write("Adjust the sliders and select your gender to see your predicted music genre.")

# Input widgets for user interaction
# st.slider: Allows users to select a numerical value within a range.
age = st.slider(
    "Select your Age:",
    min_value=18, # Minimum age
    max_value=60, # Maximum age
    value=25, # Default starting age
    step=1 # Increment step
)

# st.radio: Allows users to select one option from a list.
gender_option = st.radio("Select your Gender:", ("Male", "Female"))

# Encode gender: The model expects numerical input (1 for Male, 0 for Female).
gender_encoded = 1 if gender_option == "Male" else 0

# Create a Pandas DataFrame from the user's input.
# The model expects input in the same format as it was trained (a DataFrame with 'age' and 'gender' columns).
input_data = pd.DataFrame([[age, gender_encoded]], columns=['age', 'gender'])

# Button to trigger the prediction
if st.button("Predict My Music Genre 🎵"):
    # Perform prediction using the loaded model.
    # .predict() returns an array, so [0] extracts the single prediction.
    prediction = model.predict(input_data)[0]
    st.success(f"Based on your input, you might prefer: **{prediction}**!")

st.markdown("---")

# --- 7. Model Performance Section ---
st.header("⚙️ Model Performance Overview")
st.write("Transparency about the model's accuracy on unseen data.")

# For demonstration, we'll re-split the data here to calculate accuracy.
# In a larger, more complex application, you might save the test set or accuracy
# score from 'train_model.py' to avoid re-calculating here.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_full = music_data.drop(columns=['genre'])
y_full = music_data['genre']
# Use the same random_state as in train_model.py for consistent split
_, X_test_eval, _, y_test_eval = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Make predictions on the test set for evaluation
test_predictions = model.predict(X_test_eval)
test_accuracy = accuracy_score(y_test_eval, test_predictions)

# Display the accuracy using a Streamlit metric widget
st.metric(label="Model Prediction Accuracy", value=f"{test_accuracy*100:.2f}%")
st.caption(f"This is the accuracy of the model on the 20% of data it had not seen during training. "
           f"An accuracy of {test_accuracy*100:.2f}% means the model correctly predicted the genre for "
           f"{test_accuracy*100:.2f}% of the test cases.")

st.markdown("---")

# --- 8. Project Details and Footer ---
st.header("About This Project")
st.write(
    """
    This project serves as a practical demonstration of fundamental machine learning concepts:
    -   **Data Loading and Preprocessing:** Using `pandas` to handle tabular data.
    -   **Supervised Learning (Classification):** Implementing a `DecisionTreeClassifier` from `scikit-learn`.
    -   **Model Training and Evaluation:** Splitting data, fitting a model, and assessing its performance (`accuracy_score`).
    -   **Model Persistence:** Saving and loading trained models using `joblib`.
    -   **Interactive Web Application Development:** Building a user-friendly dashboard with `Streamlit`.
    -   **Data Visualization:** Creating informative plots with `matplotlib` and `seaborn`.

    This dashboard is part of my portfolio to showcase skills in data science and machine learning.
    """
)
st.markdown("---")
st.write("Developed by Augustine Khumalo | Connect with me on LinkedIn.")
