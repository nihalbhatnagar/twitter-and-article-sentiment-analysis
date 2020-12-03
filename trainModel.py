import numpy as np
import pandas as pd
from nltk import classify
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pickle


# Extracting word features
def get_words_in_tweets(p):
    all = []
    for (words, sentiment) in p:
        all.extend(words)
    return all


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


def wordcloud_draw(data, color='black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                             if 'http' not in word
                             and not word.startswith('@')
                             and not word.startswith('#')
                             and word != 'RT'
                             ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color=color,
                          width=2500,
                          height=2000
                          ).generate(cleaned_word)
    plt.figure(1, figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


data = pd.read_csv('C:/Users/nihal/PycharmProjects/ISMSentimentAanalysis/data/stock_data.csv')
# Keeping only the neccessary columns
data = data[['Text', 'Sentiment']]

# Splitting the dataset into train and test set
train, test = train_test_split(data, test_size=0.1)
# Removing neutral sentiments
train = train[train.Sentiment != "0"]

train_pos = train[train.Sentiment == 1]
train_pos = train_pos['Text']
train_neg = train[train.Sentiment == -1]
train_neg = train_neg['Text']

tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.Text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
                     if 'http' not in word
                     and not word.startswith('@')
                     and not word.startswith('#')
                     and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.Sentiment))

test_pos = test[test['Sentiment'] == 1]
test_pos = test_pos['Text']
test_neg = test[test['Sentiment'] == -1]
test_neg = test_neg['Text']

w_features = get_word_features(get_words_in_tweets(tweets))
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
save_classifier = open("data/naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
# classifier_f = open("data/naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()
for index, row in test.iterrows():
    words_filtered = [e.lower() for e in row.Text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
                     if 'http' not in word
                     and not word.startswith('@')
                     and not word.startswith('#')
                     and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.Sentiment))
w_features = get_word_features(get_words_in_tweets(tweets))
test_set = nltk.classify.apply_features(extract_features, tweets)
print("Accuracy is:" + str(classify.accuracy(classifier, test_set)))

print(classifier.show_most_informative_features(10))
