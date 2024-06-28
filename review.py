from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
import string
from google_play_scraper import Sort, reviews
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import nltk

# تنزيل الموارد اللازمة من NLTK
#nltk.downloader.download('vader_lexicon')
nltk.downloader.download('stopwords')
#nltk.download('stopwords')
#nltk.download('vader_lexicon')

# إعداد FastAPI
app = FastAPI()

# نموذج الإدخال
class AppReviewsRequest(BaseModel):
    app_id: str
    lang: str = 'en'
    country: str = 'us'
    count: int = 100

# نموذج الإخراج
class SentimentResponse(BaseModel):
    content: str
    content_clean: str
    score: int
    positive: float
    negative: float
    neutral: float
    compound: float
    sentiment: str

stemmer = SnowballStemmer('english')
stopword = set(stopwords.words('english'))
sentiments = SentimentIntensityAnalyzer()

# دالة تنظيف النصوص
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = [stemmer.stem(word) for word in text]
    text = " ".join(text)
    return text

# نقطة النهاية لتحليل المشاعر
@app.post("/analyze_reviews", response_model=list [SentimentResponse])
def analyze_reviews(request: AppReviewsRequest):
    en_us_reviews = reviews(
        request.app_id,
        lang=request.lang,
        country=request.country,
        sort=Sort.MOST_RELEVANT,
        count=request.count
    )

    df = pd.DataFrame(en_us_reviews[0])
    df['content_clean'] = df['content'].apply(clean)

    df['positive'] = [sentiments.polarity_scores(i)['pos'] for i in df['content_clean']]
    df['negative'] = [sentiments.polarity_scores(i)['neg'] for i in df['content_clean']]
    df['neutral'] = [sentiments.polarity_scores(i)['neu'] for i in df['content_clean']]
    df['compound'] = [sentiments.polarity_scores(i)['compound'] for i in df['content_clean']]

    result = df["compound"].values
    sentiment = []

    for i in result:
        if i >= 0.05:
            sentiment.append('Positive')
        elif i <= -0.05:
            sentiment.append('Negative')
        else:
            sentiment.append('Neutral')

    df["sentiment"] = sentiment
    response = df[['content', 'content_clean', 'score', 'positive', 'negative', 'neutral', 'compound', 'sentiment']]

    return response.to_dict(orient='records')

# تشغيل الخادم
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)