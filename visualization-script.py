import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

# Load your topic model data
topics_file = "topics.txt"
topics_per_document_file = "topics_per_document.txt"

# Read topics data
with open(topics_file, "r") as f:
    topics_data = f.readlines()

# Process topics data
topics = []
for line in topics_data:
    parts = line.strip().split(": ")
    topic_words = parts[1].split(" + ")
    topic_words = [word.split("*")[1].strip('"') for word in topic_words]
    topics.append(topic_words)

# Create a dictionary
dictionary = Dictionary(topics)

# Create a corpus
corpus = [dictionary.doc2bow(topic) for topic in topics]

# Create an LDA model
lda_model = LdaModel(corpus, num_topics=len(topics), id2word=dictionary)

# Read topics per document data
with open(topics_per_document_file, "r") as f:
    topics_per_document_data = f.readlines()

# Process topics per document data
documents = []
for line in topics_per_document_data:
    parts = line.strip().split(": ")
    document_id = int(parts[0].split()[1])
    topic_probs = eval(parts[1])
    document_topics = [(topic[0], topic[1]) for topic in topic_probs]
    documents.append(document_topics)

# Create pyLDAvis visualization
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

# Save or display the visualization
pyLDAvis.save_html(vis_data, "lda_visualization.html")
