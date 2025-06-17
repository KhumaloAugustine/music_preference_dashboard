# This script creates an interactive Streamlit dashboard
# for predicting music genre preferences based on age and gender.
# It includes enhanced data analysis, user interactions, prediction confidence,
# and dynamically updating graphs based on user filters.

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
    st.stop() # Stop the app if model files is missing

# --- 3. Dashboard Introduction ---
st.write(
    """
    Welcome to the **Enhanced Music Preference Predictor**! This interactive dashboard uses a **Decision Tree Machine Learning model**
    to predict a user's likely music genre preference based on their **age** and **gender**.

    **New Enhancements:**
    * **Deeper Data Insights:** Explore genre preferences across different age groups and analyze gender-specific tastes.
    * **Interactive Data Filtering:** Filter the raw data view to focus on specific age ranges and genders.
    * **Model Insights:** See which factors (age or gender) were more influential in the model's predictions.
    * **Prediction Confidence:** Understand the model's certainty for each predicted genre.
    * **Dynamic Visualizations:** All data insights graphs now update based on your filters!

    **How it works:**
    * The model was trained on a small dataset mapping age and gender to music genres.
    * You can input age and gender below to get an instant prediction.
    """
)

# --- 4. Interactive Raw Data Display ---
st.header("📊 Original Data Sample with Interactive Filters")
st.write("Explore the raw data used to train the model, filtered by your selections. **(These filters now also affect the graphs below!)**")

# Create interactive filters for the raw data display
st.subheader("Filter Data Sample:")
col1, col2 = st.columns(2) # Arrange filters in two columns for better layout

with col1:
    min_age_data, max_age_data = int(music_data['age'].min()), int(music_data['age'].max())
    age_range = st.slider(
        "Select Age Range for Filtering:",
        min_value=min_age_data,
        max_value=max_age_data,
        value=(min_age_data, max_age_data), # Default to full range
        step=1
    )
with col2:
    selected_genders = st.multiselect(
        "Select Gender(s) for Filtering:",
        options=['Male', 'Female'],
        default=['Male', 'Female'] # Default to both genders selected
    )

# Apply filters to the data
filtered_data = music_data[
    (music_data['age'] >= age_range[0]) &
    (music_data['age'] <= age_range[1]) &
    (music_data['gender_label'].isin(selected_genders))
].copy() # Use .copy() to avoid SettingWithCopyWarning when adding temporary columns

# Display the filtered data
st.dataframe(filtered_data)
st.caption(f"Showing {len(filtered_data)} of {len(music_data)} total entries based on your filters.")
st.markdown("---")

# --- 5. Enhanced Exploratory Data Analysis (EDA) - Visualizations ---
st.header("📈 Dynamic Data Insights and Distributions")
st.write("These visualizations update automatically based on the filters you applied above. (Showing data for **"
         f"{age_range[0]}-{age_range[1]} years** and **{', '.join(selected_genders)}**).")

if filtered_data.empty:
    st.warning("No data matches the selected filters. Please adjust your selections.")
else:
    # Plot 1: Music Genre Distribution
    st.subheader("1. Distribution of Music Genres")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.countplot(y='genre', data=filtered_data, order=filtered_data['genre'].value_counts().index, palette='viridis', ax=ax1)
    ax1.set_title('Count of Each Music Genre in Filtered Dataset')
    ax1.set_xlabel('Number of Entries')
    ax1.set_ylabel('Music Genre')
    plt.tight_layout()
    st.pyplot(fig1)

    # Plot 2: Age Distribution
    st.subheader("2. Distribution of Ages")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_data['age'], bins=min(5, len(filtered_data.index)), kde=True, color='skyblue', ax=ax2)
    ax2.set_title('Distribution of Ages in Filtered Dataset')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig2)

    # Plot 3: Genre Preference by Gender
    st.subheader("3. Music Genre Preference by Gender")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.countplot(y='genre', hue='gender_label', data=filtered_data, palette='coolwarm', ax=ax3)
    ax3.set_title('Music Genre Preference by Gender in Filtered Dataset')
    ax3.set_xlabel('Number of Entries')
    ax3.set_ylabel('Music Genre')
    ax3.legend(title='Gender')
    plt.tight_layout()
    st.pyplot(fig3)

    # New Plot 4: Genre Distribution by Age Group
    st.subheader("4. Music Genre Distribution Across Age Groups")
    # Define age bins for grouping based on the range of filtered data
    # Create bins dynamically or use a fixed set that covers a wide range
    min_age_filtered, max_age_filtered = filtered_data['age'].min(), filtered_data['age'].max()
    dynamic_bins = np.arange(min_age_filtered, max_age_filtered + 5, 5) # Bins every 5 years
    if len(dynamic_bins) < 2: # Ensure at least two bins if data is very narrow
        dynamic_bins = np.array([min_age_filtered, max_age_filtered + 1]) # Simple bin for very narrow range

    dynamic_labels = [f"{int(b)}-{int(dynamic_bins[i+1])-1}" for i, b in enumerate(dynamic_bins[:-1])]
    # Handle the last label if it goes beyond max_age_filtered slightly
    if len(dynamic_labels) > 0 and max_age_filtered >= dynamic_bins[-2]:
         dynamic_labels[-1] = f"{int(dynamic_bins[-2])}+"

    # Avoid error if only one bin is created (e.g., if max_age_filtered - min_age_filtered is very small)
    if len(dynamic_bins) > 1:
        filtered_data['age_group'] = pd.cut(filtered_data['age'], bins=dynamic_bins, labels=dynamic_labels, right=False, include_lowest=True)

        fig4, ax4 = plt.subplots(figsize=(12, 7))
        sns.countplot(y='genre', hue='age_group', data=filtered_data, palette='tab10', ax=ax4)
        ax4.set_title('Music Genre Count by Age Group in Filtered Dataset')
        ax4.set_xlabel('Number of Entries')
        ax4.set_ylabel('Music Genre')
        ax4.legend(title='Age Group')
        plt.tight_layout()
        st.pyplot(fig4)
        filtered_data.drop(columns=['age_group'], inplace=True, errors='ignore') # Clean up temporary column
    else:
        st.info("Not enough age variation in filtered data to create age groups for this plot.")


    # New Plot 5: Gender Distribution per Genre
    st.subheader("5. Gender Composition for Each Music Genre")
    # Only proceed if there's diversity in genres or genders in the filtered data
    if len(filtered_data['genre'].unique()) > 1 or len(filtered_data['gender_label'].unique()) > 1:
        genre_gender_counts = filtered_data.groupby(['genre', 'gender_label']).size().unstack(fill_value=0)
        # Ensure sum is not zero before dividing to avoid warnings/errors
        genre_gender_proportions = genre_gender_counts.apply(lambda x: x / x.sum() if x.sum() > 0 else x, axis=1)

        fig5, ax5 = plt.subplots(figsize=(12, 7))
        genre_gender_proportions.plot(kind='barh', stacked=True, colormap='Paired', ax=ax5)
        ax5.set_title('Proportion of Genders within Each Music Genre in Filtered Dataset')
        ax5.set_xlabel('Proportion')
        ax5.set_ylabel('Music Genre')
        ax5.legend(title='Gender')
        plt.tight_layout()
        st.pyplot(fig5)
    else:
        st.info("Not enough genre or gender diversity in filtered data to create this plot.")


st.markdown("---")

# --- 6. Interactive Prediction Interface ---
st.header("🔮 Get Your Personalized Music Genre Prediction!")
st.write("Adjust the sliders and select your gender to see your predicted music genre.")

# Input widgets for user interaction
age_predict = st.slider( # Renamed variable to avoid conflict with age_range slider
    "Select your Age for Prediction:",
    min_value=int(music_data['age'].min()), # Dynamic min age
    max_value=int(music_data['age'].max()), # Dynamic max age
    value=25, # Default starting age
    step=1 # Increment step
)

gender_option_predict = st.radio("Select your Gender for Prediction:", ("Male", "Female")) # Renamed

gender_encoded_predict = 1 if gender_option_predict == "Male" else 0

input_data = pd.DataFrame([[age_predict, gender_encoded_predict]], columns=['age', 'gender'])

# Button to trigger the prediction
if st.button("Predict My Music Genre 🎵"):
    # Perform prediction using the loaded model.
    prediction = model.predict(input_data)[0]
    st.success(f"Based on your input, you might prefer: **{prediction}**!")

    # --- Prediction Confidence ---
    st.subheader("Prediction Confidence:")
    probabilities = model.predict_proba(input_data)[0]

    confidence_df = pd.DataFrame({
        'Genre': genres_list,
        'Confidence': probabilities
    }).sort_values(by='Confidence', ascending=False)

    confidence_df['Confidence'] = confidence_df['Confidence'].map(lambda x: f"{x*100:.2f}%")

    st.dataframe(confidence_df.set_index('Genre'))
    st.caption("These are the probabilities of the model predicting each genre for your input.")

st.markdown("---")

# --- 7. Model Insights and Performance Section ---
st.header("⚙️ Model Insights and Performance Overview")
st.write("Understand how the model makes decisions and its overall accuracy.")

# 7.1 Feature Importance
st.subheader("7.1 Feature Importance")
st.write("Decision Tree models assign an importance score to each feature, indicating its influence on predictions.")

feature_importances = model.feature_importances_
feature_names = ['age', 'gender']

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
_, X_test_eval, _, y_test_eval = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

test_predictions = model.predict(X_test_eval)
test_accuracy = accuracy_score(y_test_eval, test_predictions)

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
