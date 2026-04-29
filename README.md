# Spam Email Classification System

A machine learning-based web application to classify emails as spam or ham (not spam).

## Description

This project provides an intuitive Streamlit interface for email classification. It leverages multiple machine learning models including Logistic Regression, Support Vector Machines (SVM), Naive Bayes, Random Forest, and an Ensemble method to predict whether a given email or batch of emails is spam. 

## Features

- Real-time classification for single emails.
- Batch classification via CSV or Excel file uploads.
- Ensemble voting mechanism utilizing multiple models.
- Model evaluation metrics dashboard.

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Installation

1. Clone the repository or download the project files.
2. Navigate to the project directory:
   ```bash
   cd "Spam Email Classification System"
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser and navigate to the local URL provided in the terminal (usually http://localhost:8501).
3. Use the sidebar to select your preferred classification model.
4. Input email text directly or upload a dataset for batch processing.

## Project Structure

- `app.py`: Main Streamlit application file.
- `src/`: Contains source code for text preprocessing and other utilities.
- `models/`: Directory where trained models and the TF-IDF vectorizer are saved.
- `data/`: Directory for datasets (if applicable).
- `requirements.txt`: List of Python dependencies.
- `.gitignore`: Specifies intentionally untracked files to ignore.

## License

This project is licensed under the MIT License.
