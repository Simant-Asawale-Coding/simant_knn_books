import streamlit as st
import pandas as pd
import pickle
import joblib

# Load the trained KNN model

model=joblib.load('knn_books.pkl')

#Load the encoders file
encoders=joblib.load('label_encoders.pkl')


#loading the encoders individually
author_encoder = encoders['authors']
language_encoder = encoders['language_code']
publisher_encoder = encoders['publisher']

# Streamlit app title and description with HTML/CSS
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"]{
            background-image: url('https://i.guim.co.uk/img/media/77e3e93d6571da3a5d77f74be57e618d5d930430/0_0_2560_1536/master/2560.jpg?width=1900&dpr=1&s=none');
            background-size: cover; /* Ensures the image covers the whole background */
            background-position: center; /* Centers the image */
            background-repeat: no-repeat; /* Prevents the image from repeating */
            }
        .st-emotion-cache-1wmy9hl {
            display: flex;
            background-color: ivory;
            flex-direction: column;
            margin: 10px;
            flex: 1 1 0%;
}
        .main-title {
            font-size: 50px;
            color: #4A90E2;
            font-weight: 700;
            text-align: center;
        }
        .description {
            font-size: 18px;
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }
        .input-section {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
        }
        .prediction-result {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .footer {
            font-size: 14px;
            color: #777;
            text-align: center;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">📚 Book Bestseller Prediction with KNN</h1>', unsafe_allow_html=True)
st.markdown("""
    <p class="description">
    Welcome to the <strong>Book Bestseller Prediction App</strong>! 🎉
    This app uses a <strong>K-Nearest Neighbors (KNN)</strong> model to predict whether a book is likely to be a bestseller based on various features.
    Simply fill in the details below and click <strong>Predict</strong> to see the result.
    </p>
""", unsafe_allow_html=True)

# Take user inputs with a neat layout
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown("### 📊 Book Features")

col1, col2 = st.columns(2)

with col1:
    average_rating = st.slider("Average Rating", min_value=1.0, max_value=5.0, step=0.05)
    num_pages = st.slider("Number of Pages", min_value=1, max_value=2000, step=1)
    ratings_count = st.slider("Ratings Count", min_value=0, max_value=1000, step=10)
    text_reviews_count = st.slider("Text Reviews Count", min_value=0, max_value=1000, step=1)

with col2:
    selected_author = st.selectbox("Select Author", author_encoder.classes_)
    selected_language = st.selectbox("Select Language", language_encoder.classes_)
    selected_publisher = st.selectbox("Select Publisher", publisher_encoder.classes_)
    years = st.slider("Year of Publication", min_value=1990, max_value=2015, step=1)

authors_encoded = author_encoder.transform([selected_author])[0]
language_code_encoded = language_encoder.transform([selected_language])[0]
publisher_encoded = publisher_encoder.transform([selected_publisher])[0]

st.markdown('</div>', unsafe_allow_html=True)

# Prepare input for model
input_data = pd.DataFrame([[average_rating, num_pages, ratings_count, text_reviews_count,
                            authors_encoded, language_code_encoded, publisher_encoded, years]],
                          columns=['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count',
                                   'authors_encoded', 'language_code_encoded', 'publisher_encoded', 'years'])

# Make predictions using the model
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]

    # Display result based on prediction
    st.markdown("### 📝 Prediction Result")
    if prediction[0] == 1:
        st.markdown(f'<div class="prediction-result"><span style="color:green;">📈 The book is predicted to be a bestseller! Probability: {probability[1] * 100:.2f}%</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="prediction-result"><span style="color:red;">📉 The book is not predicted to be a bestseller. Probability: {probability[1] * 100:.2f}%</span></div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
    ---
    <p>**Note**: This prediction is based on historical data and the model's learned patterns.<br>
    For more accurate results, consider using additional features or more complex models.</p>
    </div>
""", unsafe_allow_html=True)
