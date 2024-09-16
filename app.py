import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import os 
import jwt
import vertexai
import base64
from wordcloud import WordCloud
from PIL import Image
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
from keboola_streamlit import KeboolaStreamlit

PROJECT = 'keboola-ai'
LOCATION = 'us-central1'
MODEL_NAME = 'gemini-1.5-pro-preview-0409'
IMAGE_PATH = os.path.dirname(os.path.abspath(__file__))
KEBOOLA_LOGO_PATH = IMAGE_PATH + "/static/keboola_logo.png"
KEBOOLA_GEMINI_PATH = IMAGE_PATH + "/static/keboola_gemini.png"
LONDON_EYE_WC_PATH = IMAGE_PATH + "/static/london_eye_wc.png"

STORAGE_API_TOKEN = st.secrets['STORAGE_API_TOKEN']
KEBOOLA_HOSTNAME = st.secrets['KEBOOLA_HOSTNAME']
CREDENTIALS = service_account.Credentials.from_service_account_info(
    jwt.decode(st.secrets['ENCODED_TOKEN'], 'keboola', algorithms=['HS256'])
)

keboola = KeboolaStreamlit(KEBOOLA_HOSTNAME, STORAGE_API_TOKEN)

st.set_page_config(layout='wide')

#@st.cache_data
#def read_data(table_name):
#    df = keboola.read_table(table_name)
#    return df

def color_for_value(value):
    if value < -0.2:
        return '#EA4335'
    elif -0.2 <= value <= 0.2:
        return '#FBBC05' 
    else:
        return '#34A853' 

def sentiment_color(sentiment):
    if sentiment == "Positive":
        return "color: #34A853"
    if sentiment == "Neutral":
        return "color: #FBBC05"
    else:
        return "color: #EA4335"

def generate(content):
    try:
        vertexai.init(project=PROJECT, location=LOCATION, credentials=CREDENTIALS)
        model = GenerativeModel(MODEL_NAME)

        config = {
            'max_output_tokens': 8192,
            'temperature': 1,
            'top_p': 0.95,
        }
        responses = model.generate_content(
            contents=content,
            generation_config=config,
            stream=True,
        )
        return "".join(response.text for response in responses)
    except Exception as e:
        st.error(f"An error occurred during content generation. Please try again.")
        return ''


logo_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(KEBOOLA_LOGO_PATH, "rb").read()).decode()}" style="width: 200px; margin-bottom: 10px;"></div>'
st.markdown(f"{logo_html}", unsafe_allow_html=True)

st.title('London Eye Reviews Sentiment Analysis')

data_path = "/data/in/tables/reviews_sentiment_final_gemini.csv"
data = pd.read_csv(data_path)
data['parsed_date'] = pd.to_datetime(data['parsed_date'], format='mixed').dt.tz_localize(None)

keywords_path = "/data/in/tables/reviews_keywords_final_gemini.csv"
keywords = pd.read_csv(keywords_path)
keywords['parsed_date'] = pd.to_datetime(keywords['parsed_date'], format='mixed').dt.tz_localize(None)

data['date'] = data['parsed_date'].dt.date
data['is_widget'] = False
keywords['date'] = keywords['parsed_date'].dt.date
keywords_filtered = keywords[~keywords['keywords'].isin(['London', 'London Eye'])]

st.markdown("<br>__Filters__", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3, gap='medium')
st.markdown("<br>", unsafe_allow_html=True)    

with col1:
    min_score, max_score = st.slider(
                'Select a range for the sentiment score:',
                min_value=-1.0, max_value=1.0, value=(-1.0, 1.0),
                key="sentiment_slider")

with col2:
    source_choices = ['All'] + data['reviewSource'].unique().tolist()
    selected_sources = st.selectbox('Filter by review source:', source_choices)

with col3:
    if not data.empty:
        min_date = data['parsed_date'].min()
        max_date = data['parsed_date'].max()
        default_date_range = (min_date, max_date)
    else:
        min_date, max_date = None, None
        default_date_range = ()

    date_range = st.date_input("Select a date range:", default_date_range, min_value=min_date, max_value=max_date)

if date_range and len(date_range) == 2:
    start_date, end_date = date_range
    data = data[(data['parsed_date'] >= pd.to_datetime(start_date)) & (data['parsed_date'] <= pd.to_datetime(end_date))]
    keywords_filtered = keywords_filtered[(keywords_filtered['parsed_date'] >= pd.to_datetime(start_date)) & (keywords_filtered['parsed_date'] <= pd.to_datetime(end_date))]
else:
    st.info("Please select both start and end dates.")
	
# Apply Filters
if selected_sources == 'All':
    filtered_data = data[(data['sentiment'] >= min_score) & (data['sentiment'] <= max_score)]
    keywords_filtered = keywords_filtered[(keywords_filtered['sentiment'] >= min_score) & (keywords_filtered['sentiment'] <= max_score)]
else:
    filtered_data = data[(data['sentiment'] >= min_score) & (data['sentiment'] <= max_score) & (data['reviewSource'] == selected_sources)]
    keywords_filtered = keywords_filtered[(keywords_filtered['sentiment'] >= min_score) & (keywords_filtered['sentiment'] <= max_score) & (keywords_filtered['reviewSource'] == selected_sources)]

col1, col2, col3 = st.columns(3, gap='medium')
with col1:
    filtered_data['color'] = filtered_data['sentiment'].apply(color_for_value)

    fig = px.histogram(
        filtered_data,
        x='sentiment',
        nbins=21,  
        title='Sentiment Score Distribution',
        color='color',
        color_discrete_map='identity'  
    )

    fig.update_layout(bargap=0.1, xaxis_title='Sentiment Score', yaxis_title='Count') 
    st.plotly_chart(fig, use_container_width=True)

with col2: 
    keyword_counts = keywords_filtered.groupby('keywords')['counts'].sum().reset_index()
    top_keywords = keyword_counts.sort_values(by='counts', ascending=True).tail(10)

    fig = px.bar(top_keywords, x='counts', y='keywords', orientation='h', title='Top 10 Keywords by Count', color_discrete_sequence=['#4285F4'])
    fig.update_layout(xaxis_title='Count', yaxis_title='Keywords')

    st.plotly_chart(fig, use_container_width=True)

with col3:
    review_source_counts  = filtered_data['reviewSource'].value_counts().reset_index()
    review_source_counts .columns = ['reviewSource', 'count']
    top_10_industries = review_source_counts .head(10)
    count = top_10_industries.shape[0]
    colors = ['#D4D4D4', '#939393']
    
    fig = px.pie(
        top_10_industries, 
        names='reviewSource', 
        values='count', 
        title=f'Distribution of Reviews',
        color_discrete_sequence=colors
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([3,2])

with col1:
    st.markdown("__Data__")
    sorted_data = filtered_data.sort_values(by='parsed_date', ascending=False)
    selected_data = st.data_editor(sorted_data[['is_widget',
                                'sentiment_category',
                                'text_in_english',                            
                                'stars',
                                'reviewSource', 
                                'date',
                                'url']].style.map(
            sentiment_color, subset=["sentiment_category"]
        ), 
                column_config={'is_widget': 'Select',
                                'sentiment_category': 'Sentiment Category',
                                'text_in_english': 'Text',
                                'stars': 'Rating',
                                'reviewSource': 'Review Source',
                                'date': 'Date',
                                'url': st.column_config.LinkColumn('URL')
                                }, height=500,
                 disabled=['sentiment_category',
                           'text_in_english',                            
                           'stars',
                           'reviewSource', 
                           'date',
                           'url'],
                use_container_width=True, hide_index=True)

def generate_wordcloud(word_freq, mask_image_path):
    colormap = mcolors.ListedColormap(['#4285F4', '#34A853', '#FBBC05', '#EA4335'])
    mask_image = np.array(Image.open(mask_image_path))

    if mask_image.dtype != np.uint8:
        mask_image = mask_image.astype(np.uint8)

    wordcloud = WordCloud(width=500, height=500, background_color=None, 
                          colormap=colormap, mask=mask_image,
                          mode='RGBA').generate_from_frequencies(word_freq)
    wordcloud_array = wordcloud.to_array()

    if wordcloud_array.dtype != np.uint8:
        wordcloud_array = wordcloud_array.astype(np.uint8)
    
    return wordcloud_array
    
summary = keywords_filtered.groupby('keywords')['counts'].sum().reset_index()
word_freq = dict(zip(summary['keywords'], summary['counts']))

# Wordcloud
with col2:
    st.markdown("__Word Eye__")
    if word_freq:    
        wordcloud = generate_wordcloud(word_freq, LONDON_EYE_WC_PATH)
        fig, ax = plt.subplots(figsize=(10, 5), frameon=False)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')    
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No keywords found to generate the word cloud.")
        
#gemini_html = f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{base64.b64encode(open(KEBOOLA_GEMINI_PATH, "rb").read()).decode()}" style="width: 60px; margin-top: 30px;"></div>'
#st.markdown(f'{gemini_html}', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: left;">
    <h4>Reply to a review with AI</h4>
</div>
""", unsafe_allow_html=True)

if 'generated_responses' not in st.session_state:
    st.session_state['generated_responses'] = {}

if selected_data['is_widget'].sum() == 1:
    selected_review = selected_data[selected_data['is_widget'] == True]['text_in_english'].iloc[0]
    review_text = selected_review if selected_review else st.warning('No review found.')
    st.write(f'_Review:_\n\n{review_text}')

    if st.button('Generate response'):
        if review_text in st.session_state['generated_responses']:
            response = st.session_state['generated_responses'][review_text]
        else:
            with st.spinner('🤖 Generating response, please wait...'):
                prompt = f"""
                You are given a review on London Eye, UK. Pretend you're a social media manager for the London Eye and write a short (3-5 sentence) response to this review. Only return the response.
                
                Review:
                {review_text}
                """
                response = generate(prompt)
                if response:
                    st.session_state['generated_responses'][review_text] = response
                else:
                    st.error("Something went wrong, please try again.")
        st.write(f"_Response:_\n\n{response}")
else:
    st.info('Select the review you want to respond to in the table above.')