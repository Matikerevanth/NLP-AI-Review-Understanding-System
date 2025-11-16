import streamlit as st
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

# Load stopwords
STOPWORDS = set(stopwords.words("english"))

# Load models
predictor = pickle.load(open("Models/model_xgb.pkl", "rb"))
scaler = pickle.load(open("Models/scaler.pkl", "rb"))
cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))

stemmer = PorterStemmer()

#  UI BACKGROUND 
def add_gradient():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to right, #4facfe, #00f2fe);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_gradient()

# Preprocessing function
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return " ".join(review)

def single_prediction(text):
    processed = preprocess_text(text)
    x_vec = cv.transform([processed]).toarray()
    x_scaled = scaler.transform(x_vec)
    y_pred = predictor.predict_proba(x_scaled).argmax(axis=1)[0]
    return "Positive" if y_pred == 1 else "Negative"

def bulk_prediction(data):
    corpus = [preprocess_text(sentence) for sentence in data["Sentence"]]
    X = cv.transform(corpus).toarray()
    X_scaled = scaler.transform(X)
    preds = predictor.predict_proba(X_scaled).argmax(axis=1)
    data["Predicted sentiment"] = ["Positive" if x == 1 else "Negative" for x in preds]
    return data


# Streamlit UI
st.title("📊 Sentiment Analysis App (Streamlit Version)")

# Single text prediction
st.header("🔍 Single Text Prediction")
user_text = st.text_area("Enter text here:")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        result = single_prediction(user_text)
        st.success(f"Prediction: **{result}**")

st.write("---")

# Bulk CSV prediction
st.header("📁 Bulk Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV with a column 'Sentence'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "Sentence" not in df.columns:
        st.error("CSV must contain a column named 'Sentence'")
    else:
        st.write("Preview of uploaded file:")
        st.dataframe(df.head())

        if st.button("Run Bulk Prediction"):
            result_df = bulk_prediction(df)
            st.success("Bulk prediction completed!")

            st.subheader("Download Results")
            st.download_button(
                label="Download Predictions CSV",
                data=result_df.to_csv(index=False),
                file_name="Predictions.csv",
                mime="text/csv"
            )

            # Pie chart of sentiment distribution
            st.subheader("📈 Sentiment Distribution Graph")
            fig, ax = plt.subplots()
            result_df["Predicted sentiment"].value_counts().plot(
                kind="pie", autopct="%1.1f%%", ax=ax, shadow=True
            )
            st.pyplot(fig)
