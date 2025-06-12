# This script creates an interactive Streamlit dashboard
# for predicting music genre preferences based on age and gender.
# It includes enhanced data analysis and user interactions.

import streamlit as st
import pandas as pd
import joblib # Used for loading the pre-trained machine learning model
import matplotlib.pyplot as plt # For creating static plots
import seaborn as sns # For enhanced data visualizations
import numpy as np # For numerical operations, e.g., creating age bins

# --- 1. Dashboard Configuration and Title ---
# Set general page configurations for the Streamlit app.
st.set_page_config(
    page_title="Enhanced Music Preference Predictor", # Updated page title
    layout="centered", # Page layout (can be "centered" or "wide")
    initial_sidebar_state="auto" # State of the sidebar (if any)
)

st.title("🎶 Enhanced Music Preference Predictor Dashboard") # Main title of the application
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
        # Ensure 'gender_label' is always available for consistent plotting
        data['gender_label'] = data['gender'].map({1: 'Male', 0: 'Female'})
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
    Welcome to the **Enhanced Music Preference Predictor**! This interactive dashboard uses a **Decision Tree Machine Learning model**
    to predict a user's likely music genre preference based on their **age** and **gender**.

    **New Enhancements:**
    * **Deeper Data Insights:** Explore genre preferences across different age groups and analyze gender-specific tastes.
    * **Interactive Data Filtering:** Filter the raw data view to focus on specific age ranges and genders.
    * **Model Insights:** See which factors (age or gender) were more influential in the model's predictions.

    **How it works:**
    * The model was trained on a small dataset mapping age and gender to music genres.
    * You can input age and gender below to get an instant prediction.
    """
)

# --- 4. Interactive Raw Data Display ---
st.header("📊 Original Data Sample with Interactive Filters")
st.write("Explore the raw data used to train the model, filtered by your selections:")

# Create interactive filters for the raw data display
st.subheader("Filter Data Sample:")
col1, col2 = st.columns(2) # Arrange filters in two columns for better layout

with col1:
    min_age, max_age = int(music_data['age'].min()), int(music_data['age'].max())
    age_range = st.slider(
        "Select Age Range:",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        step=1
    )
with col2:
    selected_genders = st.multiselect(
        "Select Gender(s):",
        options=['Male', 'Female'],
        default=['Male', 'Female'] # Default to both genders selected
    )

# Apply filters to the data
filtered_data = music_data[
    (music_data['age'] >= age_range[0]) &
    (music_data['age'] <= age_range[1]) &
    (music_data['gender_label'].isin(selected_genders))
]

st.dataframe(filtered_data) # Display the filtered data
st.caption(f"Showing {len(filtered_data)} of {len(music_data)} entries.")
st.markdown("---")

# --- 5. Enhanced Exploratory Data Analysis (EDA) - Visualizations ---
st.header("📈 Deeper Data Insights and Distributions")
st.write("Understand the characteristics and relationships within the training data through visualizations.")

# Plot 1: Music Genre Distribution
st.subheader("1. Distribution of Music Genres")
fig1, ax1 = plt.subplots(figsize=(8, 5)) # Create a matplotlib figure and axes
sns.countplot(y='genre', data=music_data, order=music_data['genre'].value_counts().index, palette='viridis', ax=ax1)
ax1.set_title('Count of Each Music Genre in Dataset')
ax1.set_xlabel('Number of Entries')
ax1.set_ylabel('Music Genre')
plt.tight_layout() # Adjust plot to prevent labels from overlapping
st.pyplot(fig1) # Display the plot in Streamlit

# Plot 2: Age Distribution
st.subheader("2. Distribution of Ages")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.histplot(music_data['age'], bins=5, kde=True, color='skyblue', ax=ax2)
ax2.set_title('Distribution of Ages in Dataset')
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')
plt.tight_layout()
st.pyplot(fig2)

# Plot 3: Genre Preference by Gender
st.subheader("3. Music Genre Preference by Gender")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.countplot(y='genre', hue='gender_label', data=music_data, palette='coolwarm', ax=ax3)
ax3.set_title('Music Genre Preference by Gender in Dataset')
ax3.set_xlabel('Number of Entries')
ax3.set_ylabel('Music Genre')
ax3.legend(title='Gender') # Add a legend for gender
plt.tight_layout()
st.pyplot(fig3)

# New Plot 4: Genre Distribution by Age Group
st.subheader("4. Music Genre Distribution Across Age Groups")
# Define age bins for grouping
age_bins = [18, 25, 30, 35, 40] # Example bins, adjust as needed based on your data
age_labels = ['18-24', '25-29', '30-34', '35-39']
music_data['age_group'] = pd.cut(music_data['age'], bins=age_bins, labels=age_labels, right=False)

fig4, ax4 = plt.subplots(figsize=(12, 7))
sns.countplot(y='genre', hue='age_group', data=music_data, palette='tab10', ax=ax4)
ax4.set_title('Music Genre Count by Age Group')
ax4.set_xlabel('Number of Entries')
ax4.set_ylabel('Music Genre')
ax4.legend(title='Age Group')
plt.tight_layout()
st.pyplot(fig4)

# New Plot 5: Gender Distribution per Genre
st.subheader("5. Gender Composition for Each Music Genre")
genre_gender_counts = music_data.groupby(['genre', 'gender_label']).size().unstack(fill_value=0)
genre_gender_proportions = genre_gender_counts.apply(lambda x: x / x.sum(), axis=1) # Calculate proportions

fig5, ax5 = plt.subplots(figsize=(12, 7))
genre_gender_proportions.plot(kind='barh', stacked=True, colormap='Paired', ax=ax5)
ax5.set_title('Proportion of Genders within Each Music Genre')
ax5.set_xlabel('Proportion')
ax5.set_ylabel('Music Genre')
ax5.legend(title='Gender')
plt.tight_layout()
st.pyplot(fig5)


# Clean up temporary columns created for plotting
music_data.drop(columns=['age_group'], inplace=True, errors='ignore') # 'gender_label' is kept for consistency in plotting
st.markdown("---")

# --- 6. Interactive Prediction Interface ---
st.header("🔮 Get Your Personalized Music Genre Prediction!")
st.write("Adjust the sliders and select your gender to see your predicted music genre.")

# Input widgets for user interaction
# st.slider: Allows users to select a numerical value within a range.
age = st.slider(
    "Select your Age:",
    min_value=int(music_data['age'].min()), # Dynamic min age
    max_value=int(music_data['age'].max()), # Dynamic max age
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

# --- 7. Model Insights and Performance Section ---
st.header("⚙️ Model Insights and Performance Overview")
st.write("Understand how the model makes decisions and its overall accuracy.")

# 7.1 Feature Importance
st.subheader("7.1 Feature Importance")
st.write("Decision Tree models assign an importance score to each feature, indicating its influence on predictions.")

# Get feature importances from the trained model
feature_importances = model.feature_importances_
feature_names = ['age', 'gender'] # Names of your input features

# Create a DataFrame for better display
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

st.dataframe(importance_df.set_index('Feature'))
st.caption("A higher 'Importance' value means the feature had more influence on the model's decisions.")

# 7.2 Model Performance (Accuracy)
st.subheader("7.2 Model Accuracy")
st.write("This shows how well the model performed on unseen data during training.")

# Re-split data for demonstration purposes (in real app, load saved metrics)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_full = music_data.drop(columns=['genre', 'gender_label']) # Exclude gender_label from X
y_full = music_data['genre']
# Use the same random_state as in train_model.py for consistent split
_, X_test_eval, _, y_test_eval = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Make predictions on the test set
test_predictions = model.predict(X_test_eval)
test_accuracy = accuracy_score(y_test_eval, test_predictions)

# Display the accuracy using a Streamlit metric widget
st.metric(label="Model Prediction Accuracy", value=f"{test_accuracy*100:.2f}%")
st.caption(f"The model achieved an accuracy of **{test_accuracy*100:.2f}%** on test data. "
           f"This indicates how often the model correctly predicted the music genre on data it had not seen during training.")

st.markdown("---")

# --- 8. Project Details and Footer ---
st.header("About This Project")
st.write(
    """
    This project demonstrates a practical application of fundamental machine learning concepts:
    -   **Data Loading and Preprocessing:** Using `pandas` to handle tabular data.
    -   **Supervised Learning (Classification):** Implementing a `DecisionTreeClassifier` from `scikit-learn`.
    -   **Model Training and Evaluation:** Splitting data, fitting a model, and assessing its performance (`accuracy_score`).
    -   **Model Persistence:** Saving and loading trained models using `joblib`.
    -   **Interactive Web Application Development:** Building an enhanced user-friendly dashboard with `Streamlit`.
    -   **Advanced Data Visualization:** Creating informative plots with `matplotlib` and `seaborn` to extract insights.

    This dashboard is part of my portfolio to showcase skills in data science and machine learning.
    """
)
st.markdown("---")
st.write("Developed by Augustine Khumalo | Connect with me on LinkedIn!(https://www.linkedin.com/in/augustine-khumalo)")
