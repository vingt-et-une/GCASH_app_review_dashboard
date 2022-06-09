import pandas as pd
import nltk
from datetime_truncate import truncate
import re 

nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json

with open('node_modules/stopwords-tl/stopwords-tl.json') as f:
  filipino_stopwords = json.load(f)
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, PorterStemmer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag


playstore = pd.read_csv('Data/gcash_playstore_reviews.csv')
appstore = pd.read_csv('Data/gcash_appstore_reviews.csv')
playstore = playstore.drop(columns='Unnamed: 0')
appstore = appstore.drop(columns='Unnamed: 0')
playstore['isgoogle'] = 1
appstore['isgoogle'] = 0

appstore = appstore.rename(columns = {'review': 'content','rating':'score','date':'at','developerResponse':'replyContent'})
all_data = pd.concat([playstore, appstore])
all_data['content'] = all_data['content'].apply(lambda x : str(x)) 
all_data['at'] = pd.to_datetime(all_data['at'])
all_data['month'] = all_data['at'].apply(lambda x: truncate(x, 'month').date())


ver_date = all_data[['reviewCreatedVersion', 'at']].sort_values('at')
ver_date['reviewCreatedVersion'] = ver_date['reviewCreatedVersion'].fillna(method='ffill')
ver_date = ver_date.rename(columns={'reviewCreatedVersion':'imputedVersions'}).sort_values('at',ascending =False)
all_data = all_data.merge(ver_date, how='left',on='at')


stopwords_list = set(stopwords.words('english')).union(set(filipino_stopwords)).union(set(['yung','nyo','di','wala','naman','nag','pera','sana']))


all_data['tokens']= all_data['content'].apply(lambda x: word_tokenize(re.sub('\W+', " ", x.lower())))
all_data['tokens'] = all_data['tokens'].apply(lambda x: [w for w in x if (not w in stopwords_list)])
all_data['tokens'] = all_data['tokens'].apply(lambda x: pos_tag(x))
all_data['tokens'] = all_data['tokens'].apply(lambda x: [word[0] for word in x if word[1] in ['NN','NNP','NNPS','NNS','VB','VBG','VBD','VBN','VBP','VBZ','JJ','JJR','JJS']])


dash_data = all_data[['score','month','tokens']]
dash_data = dash_data.groupby(['score','month'])['tokens'].agg(sum).reset_index()


all_data.to_pickle('Data/all_data.pkl')
dash_data .to_pickle('Data/dash_data.pkl')
