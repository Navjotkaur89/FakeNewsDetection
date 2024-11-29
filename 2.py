import streamlit as st
from streamlit_card import card as stc
import random, csv
import requests
import uuid
from datetime import datetime
import pytz
from googletrans import Translator
from streamlit_lottie import st_lottie
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import joblib
import pickle





print("Model loaded successfully")

# Setting up parameters (common values for fake news detection)
MAX_WORDS = 10000  # Maximum number of words to keep
MAX_LEN = 200     # Maximum length of each text

# Create tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS)


def add_custom_css():
    st.markdown("""<style>
        /* Background color */
        .main {
            background-color: #0e1117;
        }
        /* Heading font size */
        h1 {
            font-size: 60px !important;
            font-weight: bold !important;
            color: #9966cc;
        }
        h2 {
            font-size: 70px !important;
            color: #9966cc;
        }
        h3 {
            font-size: 20px !important;
            color: #bd33a4;
            text-align: right;
        }
        h4 {
            font-size: 30px !important;
            color: #ffffff;
        }
        
        /* Button styling */
        button[kind="primary"] {
            background-color: #007BFF;
            color: white;
        }

        
        </style>""", unsafe_allow_html=True)
st.cache_data()
def getNews(category='all'):
    headers = {
        'authority': 'inshorts.com',
        'accept': '*/*',
        'accept-language': 'en-GB,en;q=0.5',
        'content-type': 'application/json',
        'referer': 'https://inshorts.com/en/read',
        'sec-ch-ua': '"Not/A)Brand";v="99", "Brave";v="115", "Chromium";v="115"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    }

    if category == 'all':
        response = requests.get(
            'https://inshorts.com/api/en/news?category=all_news&max_limit=1000&include_card_data=true')
    else:
        params = (
            ('category', category),
            ('max_limit', '1000'),
            ('include_card_data', 'true')
        )
        response = requests.get(
            f'https://inshorts.com/api/en/search/trending_topics/{category}', headers=headers, params=params)
    
    try:
        news_data = response.json()['data']['news_list']
    except Exception as e:
        print(response.text)
        news_data = None

    newsDictionary = {
        'success': True,
        'category': category,
        'data': []
    }

    if not news_data:
        newsDictionary['success'] = False
        newsDictionary['error'] = 'Invalid Category'
        return newsDictionary

    for entry in news_data:
        try:
            news = entry['news_obj']
            author = news['author_name']
            title = news['title']
            imageUrl = news['image_url']
            url = news['shortened_url']
            content = news['content']
            timestamp = news['created_at'] / 1000
            dt_utc = datetime.utcfromtimestamp(timestamp)
            tz_utc = pytz.timezone('UTC')
            dt_utc = tz_utc.localize(dt_utc)
            tz_ist = pytz.timezone('Asia/Kolkata')
            dt_ist = dt_utc.astimezone(tz_ist)
            date = dt_ist.strftime('%A, %d %B, %Y')
            time = dt_ist.strftime('%I:%M %p').lower()
            readMoreUrl = news['source_url']

            newsObject = {
                'id': uuid.uuid4().hex,
                'title': title,
                'imageUrl': imageUrl,
                'url': url,
                'content': content,
                'author': author,
                'date': date,
                'time': time,
                'readMoreUrl': readMoreUrl
            }
            newsDictionary['data'].append(newsObject)
        except Exception as e:
            print(f"Error processing news entry: {entry}")
            print(e)
    
    return newsDictionary

# Function to read CSV files


# def predict_fake_news(text):
#     # Preprocess the text
#     try:
#         model_1 = keras.models.load_model('my_model_1.h5')
#         model_2 = keras.models.load_model('my_model_1.keras')
#         st.success("Models loaded successfully.")
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#     text_sequence = tokenizer.texts_to_sequences([text])
#     text_padded = pad_sequences(text_sequence, maxlen=MAX_LEN)
    
#     # Make prediction
#     prediction_1 = model_1.predict(text_padded)[0][0]
#     prediction_2 = model_2.predict(text_padded)[0][0]
    
#     return prediction_1, prediction_2

# def check():
#     st.title("Fake News Detection System")
    
#     st.write("Enter your news text below (press Enter twice to submit):")
    
#     # Text input for news article
#     news = st.text_area("Enter the news here", height=150, 
#                         placeholder="Type your news here...", 
#                         label_visibility='collapsed')  # Hide label for cleaner look

#     if st.button('Predict'):
#         if news:
#             try:
#                 # Make predictions
#                 prediction_1, prediction_2 = predict_fake_news(news)
                
#                 # Show predictions
#                 st.write("Prediction Result:")
#                 st.write(f"Model 1 Probability of being fake news: {prediction_1:.2%}")
#                 st.write(f"Model 2 Probability of being fake news: {prediction_2:.2%}")
                
#                 if prediction_1 > 0.5 or prediction_2 > 0.5:
#                     st.error("This news is likely FAKE", icon="❌")
#                 else:
#                     st.success("This news appears to be REAL", icon="✅")
#             except Exception as e:
#                 st.error(f"Error during prediction: {e}")
#         else:
#             st.warning("Please enter some news text for prediction.")

# Function to load models (use caching to avoid reloading every time)
@st.cache_resource
def load_models():
    model_1 = load_model('my_model.h5')
    model_2 = load_model('my_model.keras')
    return model_1, model_2

# Function to load tokenizer (use caching to avoid reloading every time)
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Pre-trained max length (ensure this matches your training)
MAX_LEN = 100  # Replace with your actual maxlen value

# Prediction function
def predict_fake_news(text):
    # Load models and tokenizer
    model_1, model_2 = load_models()
    tokenizer = load_tokenizer()
    
    # Preprocess the text
    text_sequence = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_sequence, maxlen=MAX_LEN)
    
    # Make predictions
    prediction_1 = (model_1.predict(text_padded) >= 0.5).astype("int32")[0][0]
    prediction_2 = (model_2.predict(text_padded) >= 0.5).astype("int32")[0][0]
    
    return prediction_1, prediction_2

# Streamlit app function
def check():
    st.title("Fake News Detection System")
    
    st.write("Enter your news text below (press Enter twice to submit):")
    
    # Text input for news article
    news = st.text_area(
        "Enter the news here",
        height=150,
        placeholder="Type your news here...",
        label_visibility='collapsed'  # Hide label for cleaner look
    )

    if st.button('Predict'):
        if news:
            try:
                # Make predictions
                prediction_1, prediction_2 = predict_fake_news(news)
                
                # Show predictions
                st.write("Prediction Result:")
                st.write(f"Model 1 Prediction: {'FAKE' if prediction_1 == 1 else 'REAL'}")
                st.write(f"Model 2 Prediction: {'FAKE' if prediction_2 == 1 else 'REAL'}")
                
                # Final decision
                if prediction_1 == 0 or prediction_2 == 0:
                    st.success("This news appears to be REAL", icon="✅")
                else:  
                    st.error("This news is likely FAKE", icon="❌")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter some news text for prediction.")


def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()






def translate_news(title, author, content, target_language):
    translator = Translator()
    
    # Translate title and content
    translated_title = translator.translate(title, dest=target_language).text
    translated_content = translator.translate(content, dest=target_language).text

    # Check if the author's name was changed by the translation
    translated_author = translator.translate(author, dest=target_language).text
    if translated_author.lower() == author.lower():  # If name remains the same
        translated_author = author  # Use original author name
    
    return translated_title, translated_author, translated_content

def show_live_news():
    st.markdown("""
                <h2 style='text-align: center; color: #9966cc; margin-top: -1px;
                text-shadow: 0 1px 0 #ccc, 0 2px 0 #c9c9c9, 0 3px 0 #bbb, 
                             0 4px 0 #b9b9b9, 0 5px 0 #aaa, 0 6px 1px rgba(0,0,0,0.1),
                             0 0 5px rgba(0,0,0,0.2), 0 1px 3px rgba(0,0,0,0.5);'>
                Live News</h2>
            """, unsafe_allow_html=True)
    # Language selection dropdown
    languages = {
        "English": "en",
        "Hindi": "hi",
        "Punjabi": "pa",
        "Spanish": "es",
        "Dutch": "nl",
        "French": "fr",
        "Tamil": "ta",
        "Korean": "ko",
        "Chinese": "zh-cn",
        "Marathi": "mr"
    }
    
    # Category selection dropdown
    categories = {
        "All": "all",
        "National": "national",
        "Business": "business",
        "Sports": "sports",
        "World": "world",
        "Politics": "politics",
        "Technology": "technology",
        "Startup": "startup",
        "Entertainment": "entertainment",
        "Miscellaneous": "miscellaneous",
        "Hatke": "hatke",
        "Science": "science",
        "Automobile": "automobile"
    }

    col1, col2 = st.columns(2)
    with col1:
        selected_language = st.selectbox("Select language for translation", list(languages.keys()), index=0)
    with col2:
        selected_category = st.selectbox("Select news category", list(categories.keys()), index=0)

    # Fetch live news data from Inshorts based on selected category
    news_data = getNews(categories[selected_category])['data']

    if not news_data:
        st.error(f"Unable to fetch live news for the {selected_category} category at the moment.")
        return

    # Display each news article with translation
    for news in news_data:
        st.image(news['imageUrl'], width=750)

        # Translate title, author, and content
        translated_title, translated_author, translated_content = translate_news(
            news['title'], news['author'], news['content'], languages[selected_language]
        )
        
        # Display translated title, author, and content
        st.markdown(f"<h4 style='font-size: 10px;'>{translated_title}</h4>", unsafe_allow_html=True)
        st.write(f"**Date**: {news['date']} - **Time**: {news['time']}")
        st.write(f"**Author**: {translated_author}")
        st.write(f"**Translated Content**: {translated_content}")
        st.markdown(f"[Read More]({news['readMoreUrl']})")
        st.markdown("---")

# Main function with sidebar for navigation





def main():
    # Apply custom CSS for background and font
    add_custom_css()
    
   
    
    # Use markdown to add a styled heading in the sidebar
    st.sidebar.markdown("""
                <h2 style='text-align: center; color: #9966cc; margin-top: -1px;
                text-shadow: 0 1px 0 #ccc, 0 2px 0 #c9c9c9, 0 3px 0 #bbb, 
                             0 4px 0 #b9b9b9, 0 5px 0 #aaa, 0 6px 1px rgba(0,0,0,0.1),
                             0 0 5px rgba(0,0,0,0.2), 0 1px 3px rgba(0,0,0,0.5);'>
                Menu</h2>
    
            """, unsafe_allow_html=True)
    
    #     Greeting the user
    # st.sidebar.markdown("<h3 style='color: #9966cc;'>Hello, User!</h3>", unsafe_allow_html=True)
    # st.sidebar.markdown("### Your source for reliable news updates.", unsafe_allow_html=True)

   # Menu with icons
    menu_options = {
        "Check News": "📰",
        "Show Live News": "🌐"
    }
    menu = st.sidebar.selectbox("Select an option", list(menu_options.keys()), format_func=lambda x: f"{menu_options[x]} {x}")


    if menu == "Check News":
        check()
        # st.write("Check News")
    elif menu == "Show Live News":
        show_live_news()

if __name__ == '__main__':
    main()