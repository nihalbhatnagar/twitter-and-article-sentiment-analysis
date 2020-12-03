import requests
import json
from bs4 import BeautifulSoup

def parse():##Takes all RSS feed data and inputs into string array
    article_list = [];
    try:

        r = requests.get('https://www.cnbc.com/id/19746125/device/rss/rss.xml')##Establish connection and retrieve site data
        soup = BeautifulSoup(r.content, features='xml')#begins parsing data and uses XML formatting
        articles = soup.findAll('item')
        for a in articles:
            title = a.find('title').text

            description = a.find('description').text
            article = {
                'title': title,
                'description': description
            }
            article_list.append(article)
        return save(article_list)
    except Exception as e:
        print('The scraping job failed. See exception: ')
        print(e)
def save(article_list):
    with open('articles.txt', 'w') as outfile:
        json.dump(article_list, outfile)
print('Starting scraping')
parse()
print('Finished scraping')