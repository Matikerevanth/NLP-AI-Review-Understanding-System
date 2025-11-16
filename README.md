# NLP-AI-Review-Understanding-System
# Sentiment Analysis Web App
 This project is a Sentiment Analysis Web Application built using Streamlit, NLTK, and a trained XGBoost Machine Learning model.
 It allows users to:
 ✔️ Predict sentiment (Positive / Negative) for a single text input
✔️ Upload a CSV file and get bulk predictions
✔️ Download the output file containing predicted sentiments
✔️ View sentiment distribution through a pie chart
✔️ Use a clean UI with gradient background

## Features
### Single Text Prediction
* Enter any text (e.g., product reviews).
* The app processes and predicts whether the sentiment is Positive or Negative.
## Bulk CSV Prediction
* Upload a CSV file containing a column named Sentence.
* The app:
-> Preprocesses all rows
-> Predicts the sentiment
-> Adds a new column Predicted sentiment
-> Allows you to download the updated CSV

 ##  Pie Chart
 * Visualizes the sentiment distribution of the uploaded CSV file.
   
## Custom UI Background
Gradient background for better UI experience.

## Model & Preprocessing
The app uses:
* XGBoost model → model_xgb.pkl
* CountVectorizer → countVectorizer.pkl
* Scaler → scaler.pkl

## Preprocessing steps:
1. Remove non-alphabetical characters
2. Convert text to lowercase
3. Remove stopwords
4. Apply stemming
5. Convert to vector using CountVectorizer
6. Scale features
7. Predict using trained ML model

## CSV Format Requirement
### Sentence
This product is amazing
Worst quality ever
Good value for money

## Technologies Used
* Python
* Streamlit
* NLTK
* Pandas
* Matplotlib
* XGBoost
* Scikit-learn





   
 
