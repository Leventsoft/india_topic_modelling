import nltk
import pandas as pd
from gensim import corpora, models

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

#need to originalize the part after here

def preprocess_text(input_text):
    input_text = str(input_text)  # Conversion of any float to a string
    tokens = word_tokenize(input_text.lower())  # Lowercase conversion and tokenization
    tokens = [tkn for tkn in tokens if tkn.isalpha()]  # Removal of non-alphabetic tokens
    tokens = [tkn.translate(translator) for tkn in tokens]  # Elimination of punctuation
    tokens = [tkn for tkn in tokens if tkn not in stop_words]  # Exclusion of stopwords
    return tokens

processed_docs = [preprocess_text(doc) for doc in english_reviews]

# Generating a dictionary representation of the processed documents
dictionary = corpora.Dictionary(processed_docs)

# Conversion of the preprocessed documents into a bag-of-words representation
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# LDA model training
num_topics = 7
lda_model = models.LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Writing topics and their associated words into a .txt file
with open('topics.txt', "w") as f:
    for topic_id, topic_words in lda_model.print_topics():
        f.write(f"Topic {topic_id}: {topic_words}\n")
 
# Retrieving assigned topics for each document
document_topics = [lda_model[doc] for doc in bow_corpus]
with open('topics_per_document.txt', "w") as g:
    for i, doc_topics in enumerate(document_topics):
        g.write(f"Document {i}: {doc_topics}\n")

