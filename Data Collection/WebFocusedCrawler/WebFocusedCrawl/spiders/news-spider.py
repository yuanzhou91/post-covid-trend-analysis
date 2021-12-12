# Scrape urls fetched from news-api, saving page html code to news-html directory 
# We scrape the text data creating a JSON lines file items.jl
# with each line of JSON representing a news article
# Subsequent text parsing of these JSON data will be needed
# Prerequisites: 
# ensure that NLTK has been installed along with the stopwords corpora
# pip install nltk
# python -m nltk.downloader stopwords

import scrapy
import re
import uuid
from datetime import date, timedelta
import os.path
from newsapi import NewsApiClient
from WebFocusedCrawl.items import WebfocusedcrawlItem  # item class 
import nltk  # used to remove stopwords from tags
import re  # regular expressions used to parse tags
from bs4 import BeautifulSoup

def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    good_tokens = [token for token in tokens if token not in stopword_list]
    return good_tokens     

class NewsSpider(scrapy.Spider):
    name = "news-spider"

    def start_requests(self):       
        newsapi = NewsApiClient(api_key='3b8aa9b78d624ef78a8af59371cad63a')
        # We set a few keywords that are equivalent when describing covid-19
        search_keywords = ['covid-19', 'coronavirus', 'SARS-CoV-2', '2019-ncov']
        # Initialize urls with set to avoid duplicate article URLs when doing searching.
        urls = set()
        urls_limit = 500
        
        # Gather the list of news sources covering technology news.
        sources_json = newsapi.get_sources(language='en', category='technology')
        sources = list(map(lambda x:x['id'], sources_json['sources']))
        
        
        # Construct a list of article URLs to scrape from.
        for keyword in search_keywords:
            # Collect latest 30 days of news
            curr_date = date.today()
            for k in range(1, 7):
                from_date = curr_date - timedelta(days=5)
                to_date = curr_date
                # We get first 5 pages from the articles search result by each keyword.
                for i in range(1, 6):
                    try:
                        all_articles = newsapi.get_everything(q=keyword,
                                                  from_param=from_date.strftime("%Y-%m-%d"),
                                                  to=to_date.strftime("%Y-%m-%d"),
                                                  language='en',
                                                  sort_by='relevancy',
                                                  sources = ','.join(sources),
                                                  page=i)
                        urls = urls.union(list(map(lambda x:x['url'], all_articles['articles'])))
                        if len(urls) > urls_limit:
                            break
                    except:
                        continue
                if len(urls) > urls_limit:
                    break
                curr_date = from_date    
            if len(urls) > urls_limit:
                break
        # Scrape the website for each of the url links to.
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self, response):
        # first part: save news articles page html to news-html directory
        page = response.url.strip("/").strip(".html").split("/")[-1]
        page_dirname = 'news-html'
        filename = '%s.html' % page
        with open(os.path.join(page_dirname,filename), 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename) 

        # second part: extract text for the item for document corpus
        item = WebfocusedcrawlItem()
        # Generate uuid of 8 characters length as unique identifier for the article.
        item['id'] = str(uuid.uuid4())[:8]
        item['url'] = response.url
        # First non-empty h1 text after stripping non-words.
        titlelist = list(map(lambda x: x.strip(), response.css('h1::text').extract())) 
        item['title'] = next(s for s in titlelist if s)

        # Because each website's article body is not under same xpath, so we must first extract all text from the website,
        # and do some further cleaning works.
        soup = BeautifulSoup(response.body)
        raw_html_text = soup.get_text(" ")
        # We need to process the text to make it clean and free of punctuations
        # Leave only English words using ntlk
        words = set(nltk.corpus.words.words())
        words_text = " ".join(w for w in nltk.wordpunct_tokenize(raw_html_text) if w.lower() in words or not w.isalpha()) 
        # Remove punctuations using regex
        cleaned_full_article = re.sub(r'[^\w\s]', ' ', words_text)
        # Get first 500 words, split by whitespaces
        split_words = [x.strip() for x in cleaned_full_article.split(' ')]
        split_non_empty_words = [ x for x in split_words if len(x) > 0]
        item['body'] = " ".join(split_non_empty_words[:500]).strip()

        return item 