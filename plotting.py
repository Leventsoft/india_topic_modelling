import matplotlib.pyplot as plt
import re

# Read the topics from the topics.txt file
topics = []
with open('topics.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        _, topic_words = line.strip().split(':')
        topics.append(topic_words.strip())

# Count the number of documents for each topic
topic_counts = [0] * len(topics)
with open('topics_per_document.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        doc_topics = re.findall(r'\d+', line)  # Extract topic IDs using regex
        for topic_id in doc_topics:
            topic_counts[int(topic_id)] += 1

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(range(len(topics)), topic_counts, tick_label=topics, color='skyblue')
plt.xticks(rotation=90)
plt.xlabel('Topics')
plt.ylabel('Number of Documents')
plt.title('Topic Distribution in Documents')
plt.tight_layout()
plt.show()