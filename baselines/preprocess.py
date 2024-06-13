import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string

nltk.download('punkt')
nltk.download('stopwords')

def summarize_article(article, num_sentences=3):
    # Tokenize the article into sentences
    sentences = sent_tokenize(article)

    # Tokenize the article into words
    words = word_tokenize(article.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and word not in string.punctuation]

    # Get the frequency of each word
    word_freq = Counter(words)

    # Score each sentence based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        score = sum(word_freq[word] for word in sentence_words if word in word_freq)
        sentence_scores[sentence] = score

    # Select the top N sentences
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    # Join the top sentences to form the summary
    summary = ' '.join(top_sentences)

    return summary

def create_vocabulary(queries):
    stop_words = set(stopwords.words('english'))
    vocabulary = set()

    for query in queries:
        # Tokeniser les mots de la query
        words = word_tokenize(query.lower())

        # Enlever les stopwords et la ponctuation
        words = [word for word in words if word not in stop_words and word not in string.punctuation]

        # Ajouter les mots à l'ensemble de vocabulaire
        vocabulary.update(words)

    return vocabulary

def reduce_articles_to_vocabulary(article, vocabulary):
        # Tokenize the article into words
        words = article.split()

        # Filter out words not in the vocabulary
        reduced_words = set([word for word in words if word in vocabulary])

        reduced_article = ' '.join(reduced_words)

        # Append the reduced article to the list

        return reduced_article