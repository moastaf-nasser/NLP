# NLP

Functionality:

This code builds a FastAPI application that performs sentiment analysis on app reviews retrieved from the Google Play Store. It allows users to submit requests specifying the app ID, language (defaults to English), country (defaults to US), and desired number of reviews (defaults to 100). The API then analyzes the reviews and returns a list of sentiment analysis results for each review.

Code Breakdown:

Imports:

fastapi: Web framework for building APIs
pydantic: Data validation for request and response models
pandas: Data manipulation library
numpy: Used for numerical operations (not used extensively here)
re: Regular expressions for text cleaning
string: Module containing string constants like punctuation
google_play_scraper: Library to scrape reviews from Google Play
nltk.corpus.stopwords: Provides stop words for text cleaning
nltk.stem.snowball.SnowballStemmer: Used for stemming words (reducing them to their base form)
nltk.sentiment.vader.SentimentIntensityAnalyzer: Sentiment analysis tool
NLTK Resource Download (Commented Out):

The commented lines download the Vader lexicon and stop words from NLTK. Run them once if you haven't already.
FastAPI App Setup:

app = FastAPI(): Creates a FastAPI application instance
Request and Response Models:

AppReviewsRequest: Defines the expected format for user requests
app_id: Required string, the ID of the app on Google Play
lang: Optional string, language of the reviews (defaults to 'en')
country: Optional string, country code for reviews (defaults to 'us')
count: Optional integer, number of reviews to retrieve (defaults to 100)
SentimentResponse: Defines the format of the response for each review
content: Original review text
content_clean: Cleaned review text after preprocessing
score: Not used in this code, could represent a custom sentiment score
positive, negative, neutral: Sentiment scores from VADER
compound: Overall sentiment score from VADER
sentiment: Categorized sentiment (Positive, Negative, Neutral)
Text Cleaning Function (clean):

Takes a text string as input
Converts it to lowercase
Removes bracketed text (e.g., https://www.youtube.com/)
Removes URLs and HTML tags
Removes punctuation
Removes newlines
Removes words containing numbers (e.g., "phone123")
Removes stop words (common words like "the", "a")
Stems words (e.g., "running" becomes "run")
Returns the cleaned text
analyze_reviews Function (API Endpoint):

Takes an AppReviewsRequest object as input
Uses google_play_scraper to retrieve reviews based on the request parameters
Converts the list of reviews to a pandas DataFrame
Applies the clean function to each review in the content column, storing the result in a new content_clean column
Uses VADER to calculate sentiment scores (positive, negative, neutral, compound) for each cleaned review and adds them to the DataFrame
Creates a new sentiment column based on sentiment thresholds (-0.05 for negative, 0.05 for positive)
Selects relevant columns from the DataFrame (content, content_clean, score, sentiment scores, and sentiment)
Converts the selected data to a list of dictionaries, each representing a single review's sentiment analysis result
Returns the list of dictionaries as the API response
Server Startup:

Conditional statement (if __name__ == "__main__":) ensures this code only runs when the script is executed directly
Imports uvicorn, a web server for running FastAPI applications
Starts the FastAPI app using uvicorn.run with the specified host (0.0.0.0 for all interfaces) and port (8000)
In essence, this code provides a sentiment analysis API for app reviews. Users can send requests to analyze reviews for a specific app, and the API returns the sentiment breakdown for each review.
