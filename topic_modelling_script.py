import pandas as pd

#import the csv dataset into a pandas dataframe
data = pd.read_csv('amazon-cellphones.csv')

#drop the unverified comments
for x in data.index:
    if data.loc[x, "verified"] == "false":
        data.drop(x, inplace=True)

#get the verified comments as a list
all_reviews = data['body'].tolist()

#get rid of non-english comments
from langdetect import detect

# Function to check if a review is in English
def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False
    
# create a list with English and verified reviews only
english_reviews = [review for review in all_reviews if is_english(review)]

#its working up until this point

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import string

stop_words = set(stopwords.words('english'))
translator = str.maketrans('', '', string.punctuation)