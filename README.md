# Flask Sentiment Analysis Web App

This project is a web application that analyzes the sentiment of text. It uses a machine learning model trained on the IMDb movie review dataset to classify a given sentence as either "Positive" or "Negative". The backend is built with Python and Flask, and the model is trained using Scikit-learn.

## Features

* **Simple Web Interface:** A clean UI to enter text and view the sentiment result.
* **Machine Learning Backend:** Uses a trained Logistic Regression model to make predictions.
* **REST API:** The Flask application exposes a `/predict` endpoint to get sentiment predictions.
* **Modular Structure:** The project is organized into separate files for training, application logic, and user interface.

## Technologies Used

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn, Pandas, Joblib
* **Frontend:** HTML, CSS, JavaScript
* **Dataset:** [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npaths/imdb-dataset-of-50k-movie-reviews)

## Project Structure

```
.
├── app.py              # Main Flask application file
├── train.py            # Script to train the model and save artifacts
├── model.pkl           # Saved trained model
├── vectorizer.pkl      # Saved TF-IDF vectorizer
├── IMDB Dataset.csv    # The dataset for training (should be downloaded)
├── .gitignore          # Specifies files for Git to ignore
├── templates/
│   └── index.html      # Frontend HTML file
└── static/
    └── style.css       # CSS for styling the frontend
```

## Setup and Usage

Follow these steps to run the project on your local machine.

### 1. Prerequisites

* Python 3.x installed
* Git installed (optional, for version control)

### 2. Set Up the Project

Clone the repository or ensure all the project files are in a single directory.

### 3. Create a Virtual Environment

It is highly recommended to use a virtual environment to keep project dependencies isolated.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python libraries.

```bash
pip install pandas scikit-learn flask joblib
```

### 5. Download the Dataset

* Download the [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npaths/imdb-dataset-of-50k-movie-reviews) from Kaggle.
* Place the `IMDB Dataset.csv` file in the root directory of your project.

### 6. Train the Model

Run the training script once to generate the model artifacts (`model.pkl` and `vectorizer.pkl`).

```bash
python train.py
```

### 7. Run the Flask Application

Start the web server.

```bash
python app.py
```

The application will be running at **http://127.0.0.1:5000/**. Open this URL in your web browser to use the Sentiment Analyzer.
