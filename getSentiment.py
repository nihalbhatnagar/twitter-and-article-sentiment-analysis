import pickle
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import io
import tweepy

tickers = ['AMZN', 'TSLA', 'GOOG', 'AAPL', 'MSFT', 'FB', 'GOOGL', 'JNJ']


def extract_features(document):
    document_words = set(document)
    features = {}

    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


def get_words_in_tweets(p):
        all = []
        for (words) in p:
            all.extend(words)
        return all



def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    print(features)
    return features


def get_tweets(query, count=300):
    consumer_key = 'sz6x0nvL0ls9wacR64MZu23z4'
    consumer_secret = 'ofeGnzduikcHX6iaQMqBCIJ666m6nXAQACIAXMJaFhmC6rjRmT'
    access_token = '854004678127910913-PUPfQYxIjpBWjXOgE25kys8kmDJdY0G'
    access_token_secret = 'BC2TxbhKXkdkZ91DXofF7GX8p2JNfbpHqhshW1bwQkgxN'
    # create OAuthHandler object
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # set access token and secret
    auth.set_access_token(access_token, access_token_secret)
    # create tweepy API object to fetch tweets
    api = tweepy.API(auth)
    # empty list to store parsed tweets
    tweets = []
    target = io.open("data/tweets.txt", 'w', encoding='utf-8')
    # call twitter api to fetch tweets
    parsed_tweets = []
    for ticker in tickers:
        q = str(query)
        a = str(q + ticker)
        fetched_tweets = api.search(a, count=count)
        # parsing tweets one by one
        for tweet in fetched_tweets:

            # empty dictionary to store required params of a tweet
            parsed_tweet = {}
            # saving text of tweet
            parsed_tweet['text'] = tweet.text
            if "http" not in tweet.text:
                line = re.sub("[^A-Za-z]", " ", tweet.text)
                target.write(line + "\n")
                parsed_tweets.append([ticker, line])
    return parsed_tweets


def get_articles():
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    f = open('data/articles.csv', 'w')

    for ticker in tickers:
        url = finwiz_url + ticker
        req = Request(url=url, headers={'user-agent': 'my-app/-2.0.1'})
        response = urlopen(req)
        # Read the contents of the file into 'html'
        html = BeautifulSoup(response, features="lxml")
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = html.find(id='news-table')
        # Add the table to our dictionary
        news_tables[ticker] = news_table

    parsed_news = []

    # Iterate through the news
    for file_name, news_table in news_tables.items():
        # Iterate through all tr tags in 'news_table'
        for x in news_table.findAll('tr'):
            # read the text from each tr tag into text
            # get text from a only
            text = x.a.get_text()
            # split text in the td tag into a list
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element
            if len(date_scrape) == 1:
                time = date_scrape[0]
            # else load 'date' as the 1st element and 'time' as the second
            else:
                date = date_scrape[0]
                time = date_scrape[1]
            # Extract the ticker from the file name, get the string up to the 1st '_'
            ticker = file_name.split('_')[0]
            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([ticker, date, time, text])
            f.write(text + "\n")
    f.close()
    return parsed_news


def sentimentAnalysisArticlesVader():
    vader = SentimentIntensityAnalyzer()
    articles = get_articles()
    # Set column names
    columns = ['ticker', 'date', 'time', 'headline']
    # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
    parsed_and_scored_news = pd.DataFrame(articles, columns=columns)
    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)
    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
    # Convert the date column from string to datetime
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
    parsed_and_scored_news.head()
    parsed_and_scored_news.to_csv('data/articleOpinionValsVader.csv')


def sentimentAnalysisTweetsVader():
    vader = SentimentIntensityAnalyzer()
    tweets = get_tweets(query="", count=19999)
    # Set column names
    columns = ['ticker', 'text']
    # Convert the parsed_tweets list into a DataFrame called 'parsed_and_scored_tweets'
    parsed_and_scored_tweets = pd.DataFrame(tweets, columns=columns)
    # Iterate through the tweets and get the polarity scores using vader
    scores = parsed_and_scored_tweets['text'].apply(vader.polarity_scores).tolist()
    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)
    # Join the DataFrames of the tweets and the list of dicts
    parsed_and_scored_tweets = parsed_and_scored_tweets.join(scores_df, rsuffix='_right')
    # Convert the date column from string to datetime
    parsed_and_scored_tweets.head()
    parsed_and_scored_tweets.to_csv('data/twitterOpinionValsVader.csv')


def sentimentAnalysisArticles():
    classifier_f = open("data/naivebayes.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    articles = get_articles()
    # Set column names
    columns = ['ticker', 'date', 'time', 'headline']
    # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
    parsed_and_scored_news = pd.DataFrame(articles, columns=columns)
    # Iterate through the headlines and get the polarity scores using vader
    # scores = parsed_and_scored_news['headline'].apply(classifier.classify).tolist()
    headlines = parsed_and_scored_news['headline'];
    scores = []
    stopwords_set = set(stopwords.words("english"))
    words = []
    for index, row in parsed_and_scored_news.iterrows():
        words_filtered = [e.lower() for e in row.headline.split() if len(e) >= 3]
        words_cleaned = [word for word in words_filtered
                         if 'http' not in word
                         and not word.startswith('@')
                         and not word.startswith('#')
                         and word != 'RT']
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
        words.append(words_without_stopwords)
    print(words)
    global w_features
    w_features = get_word_features(get_words_in_tweets(words))
    for obj in headlines:
        scores.append([obj, classifier.classify(extract_features(obj.split()))])
    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)
    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
    # Convert the date column from string to datetime
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
    parsed_and_scored_news.head()
    parsed_and_scored_news.to_csv('data/articleOpinionVals.csv')


def sentimentAnalysisTweets():
    classifier_f = open("data/naivebayes.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    tweets = get_tweets(query="", count=19999)
    # Set column names
    columns = ['ticker', 'text']

    # Convert the parsed_tweets list into a DataFrame called 'parsed_and_scored_tweets'
    parsed_and_scored_tweets = pd.DataFrame(tweets, columns=columns)
    text = parsed_and_scored_tweets['text'];
    scores = []
    stopwords_set = set(stopwords.words("english"))
    words = []
    for index, row in parsed_and_scored_tweets.iterrows():
        words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
        words_cleaned = [word for word in words_filtered
                         if 'http' not in word
                         and not word.startswith('@')
                         and not word.startswith('#')
                         and word != 'RT']
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
        words.append(words_without_stopwords)
    print(words)
    global w_features
    w_features = get_word_features(get_words_in_tweets(words))
    for obj in text:
        scores.append([obj, classifier.classify(extract_features(obj.split()))])
    # Iterate through the tweets and get the polarity scores using vader
    # scores = parsed_and_scored_tweets['text'].apply(classifier.classify).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_tweets = parsed_and_scored_tweets.join(scores_df, rsuffix='_right')

    # Convert the date column from string to datetime

    parsed_and_scored_tweets.head()
    parsed_and_scored_tweets.to_csv('data/twitterOpinionVals.csv')

global w_features
w_features = []
sentimentAnalysisTweetsVader()
sentimentAnalysisArticlesVader()
sentimentAnalysisTweets()
sentimentAnalysisArticles()
