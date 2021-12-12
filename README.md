# Post-COVID Tech Trend Analysis via NLP
This is a NLP project to reveal the hottest technologies in a post COVID-19 world, from the source of articles and news
scraped from tech media like TechCrunch, TechRadar, and CNet, etc.

## Data Collection
Collected the raw data set via development of a sophisticated web-crawler on Scrapy framework, which crawled the tech news
websites and ranked the crawled pages by applying calculating eigenvector centrality of references, and applied text cleaning
to focus on English-based articles with maximum of first 1,000 words kept.

## Data Vectorization
Vectorized the dataset by applying and comparing Doc2vec/TF-IDF and fine-tuning the dimensions to generate intermediate
data set for ML models.

## Topic Modeling
Trained a Clustering model (K-Means) to generate meaningful clusters, and applied topic modeling via LDA and bi-clustering
via Spectral Coclustering to generate top terms for each cluster. The best K was manually picked after applying t-SNE for
multidimensional scaling to 2D and visualizing the terms distribution as a scatter diagram.

Used the top terms revealed by topic modeling to create an ontology for trending tech terms, and it was a reasonable
reflection of tech trend human could sense, e.g. remote working, virtual entertainment, and social network, et
