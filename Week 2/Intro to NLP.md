# Week 2 - Introduction to Natural Language Processing

## Welcome to Week 2! 🚀

Welcome to the second week of Image captioning journey! This week, we shift gears and dive into something truly fascinating:

> **Natural Language Processing (NLP)** — teaching machines how to understand human language.

Unlike images, language is messy, ambiguous, and full of context. Yet, this is exactly what makes NLP powerful and exciting. By the end of this week, you’ll start seeing text not just as words, but as **data that can be modeled, learned from, and reasoned about**.

_________________________________________________________________________________________________

## What is Natural Language Processing(NLP)?
NLP is analysis or generation of natural language text using computers, for example:
* Machine Translation
* spell check (Autocorrect)
* Automated query answering
* Speech parsing

NLP is based on:
* Probability and statistics
*  Machine learning
* Linguistics
* Common Sense

**Why do NLP?**

Language is one of the defining characteristics of our species and a large body of knowledge can be organized and easily accessed using NLP. (Original conception of the Turing test was based on NLP)

**Some Standard Terms in NLP**
1. ***Corpus***: A collection of text samples (Documents)
2. ***Document***: A text sample
3. ***Vocabulary***: A list of words in corpus
4. ***Language Model***: How the words are supposed to be organized

 **Examples of NLP Tasks**
* Corpus level: Extracting documents
* Document level: Extracting sentence, Sentiment analysis, topic extraction, 
* Sentence Level: Text synthesis (e.g. translation, Q&A), Parsing (e.g. chunking, chinking, syntax tree),  Part of Speech tagging, Named Entity Recognition
* Tokens:  Normalisation, stemming, lemma forms of words

**Challanges in NLP**
* Large Vocabulary size
* Multiple meanings
* Synonymns
* Sarcasm, jokes, idoms, figure of speech
* Different forms of Word
* Fluid style and usuage

_________________________________________________________________________________________________

## Preprocessing in NLP
Text preprocessing in NLP is the process of transforming raw, unstructured text into a clean and structured form that machine learning or deep learning models can effectively understand. **It is a pipeline, and not every step is required for every task—choices depend on the problem, data size, and model architecture.**
### 1. Text Cleaning
Removal of Unwanted or Noisy elements from the text.

Common Operations:
* Removing HTML/XML tags
* Removing URLs, email IDs
* Removing special characters and punctuation
* Removing numbers (task-dependent)
* Removing Extra whitespaces
### 2. Case Normalisation
Converts all characters to a common case (usually lowercase), this **helps to reduce the vocabulary size and avoid treating same word as different**.
```
"Machine Learning" → "machine learning"
```
### 3. [Tokenisation](https://www.geeksforgeeks.org/nlp/nlp-how-tokenizing-text-sentence-words-works/)
Spliting text into smaller units called **tokens.**

**Types of Tokenisation:**
1. Word Tokenisation:
   ```
   "I love Machine Learning" → ["I", "love", "Machine", "Learning"]
   ```
2. Sentence Tokenisation: Used in summarization and translation.
3. Subword Tokenisation: Used in Transformer models like BERT, GPT

### 4. [Stopword Removal](https://www.geeksforgeeks.org/nlp/removing-stop-words-nltk-python/)
Removes commonly occurring words that add little semantic meaning.

Examples of stopwords: is, the, and, of, to

> **Note:** Stopword removal is useful for classical Models(TF-IDF, BoW) and is task dependent, and often **not removed** for Transformer Based Models

### 5. [Stemming](https://www.geeksforgeeks.org/machine-learning/introduction-to-stemming/)
Stemming is an important text-processing technique that reduces words to their base or root form by removing prefixes and suffixes. This process standardizes words which helps to improve the efficiency and effectiveness of various natural language processing (NLP) tasks.

This is important in the early stages of NLP tasks where words are extracted from a document and tokenized (broken into individual words).
It helps in tasks such as ***text classification, information retrieval and text summarization*** by reducing words to a base form. While it is effective, it can sometimes introduce drawbacks including potential inaccuracies and a reduction in text readability.

### 6. [Lemmatization](https://www.geeksforgeeks.org/machine-learning/python-lemmatization-approaches-with-examples/)
Lemmatization is the **process of reducing words to their base or dictionary form (lemma). Unlike stemming which simply cut off word endings, it uses a full vocabulary and linguistic rules to ensure accurate word reduction.**

For example:
* better → good
* was → be
* privatization → private

### 7. Converting Words to Vectors (Word Embeddings)
#### 1. **Bag of Words:**
A Bag-of-Words (BoW) model is a simple text representation in NLP that treats text as an unordered collection (a "bag") of its words, focusing only on word frequency, not grammar or order, to convert text into numerical features (vectors) for machine learning tasks like classification or sentiment analysis. It works by creating a vocabulary of unique words from a text corpus, then counting how many times each word appears in a given document to form a numerical vector. 

**How it Works:**
1. **Tokenization**: Break text into individual words (tokens).
2. **Vocabulary Building**: Create a list of all unique words across all documents (corpus).
3. **Vectorization**: For each document, create a vector where each position corresponds to a word in the vocabulary, and the value is the count (frequency) of that word in the document.

**Key Characteristics:**
* **simplicity:** Easy to understand and implement.
* **Disregards Order:** Ignores grammar, syntax, and word sequence, hence the "bag" analogy.
* **Captures Multiplicity:** Counts how many times a word appears (e.g., "the" appearing 3 times).
* **Feature Extraction:** Converts text into numerical data (vectors) that ML models can process.

**Learning Resources**
* [An Introduction to Bag of Words (BoW)](https://www.mygreatlearning.com/blog/bag-of-words/)
* [An Introduction to Bag of Words (BoW) (Medium Blog)](https://medium.com/@vamshiprakash001/an-introduction-to-bag-of-words-bow-c32a65293ccc)

#### 2. **Term Frequency-Inverse Document Frequency(TF-IDF):**
TF-IDF is a text mining technique that scores word importance in a document relative to a whole collection (corpus), balancing how often a word appears in one text (Term Frequency - TF) with how rare it is across all texts (Inverse Document Frequency - IDF), giving high scores to unique, relevant keywords for tasks like search engines, text classification, and topic modeling.

**How it works:**
1. **Term Frequency(TF):** Measures how often a term appears in a specific document. More occurrences generally mean higher importance in that document.
2. **Inverse Document Frequency (IDF):** Measures how rare a term is across the entire document collection. Measures how rare a term is across the entire document collection.
3. **TF-IDF Score:** Calculated by multiplying TF by IDF. A high TF-IDF score signifies a word that's frequent in a specific document but infrequent overall, making it a strong indicator of that document's topic.

**Key Benefits**
* **Identifies Keywords:** Highlights words that are truly descriptive of a document, not just common ones.
* **Text Representation:** Converts text into numerical vectors for machine learning models.
* **Information Retrieval:** Ranks search results by relevance.
* **Easy to Implement:** A relatively simple yet powerful starting point for many NLP tasks.

**Learning Resources**
* [Understanding TF-IDF: GFG](https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/)
* [TF-IDF in NLP: Medium Blog](https://medium.com/@abhishekjainindore24/tf-idf-in-nlp-term-frequency-inverse-document-frequency-e05b65932f1d)

#### 3. **Label Encoding, One hot Encoding and Word Embeddings**
* [Label Encoding ](https://www.geeksforgeeks.org/machine-learning/ml-label-encoding-of-datasets-in-python/)
* One Hot Encoding:
  1. [GFG article](https://www.geeksforgeeks.org/machine-learning/ml-label-encoding-of-datasets-in-python/)
  
  2. [One Hot Encoding and How to Implement It in Python](https://www.datacamp.com/tutorial/one-hot-encoding-python-tutorial)

  3. [One-Hot, Label, Target and K-Fold Target Encoding (YouTube)](https://youtu.be/589nCGeWG1w?si=XYoRQbJgtvA-Mgyx)

* [Word Embedding and Word2Vec, Clearly Explained!!!](https://www.youtube.com/watch?v=viZrOnJclY0)
* [A Dummy’s Guide to Word2Vec](https://medium.com/@manansuri/a-dummys-guide-to-word2vec-456444f3c673)
* [The Illustrated Word2vec - A Gentle Intro to Word Embeddings in Machine Learning](https://www.youtube.com/watch?v=ISPId9Lhc1g)

_____________________________________________________________________________________________
## Regular Expression 
Regular expressions (regex) are a powerful tool for pattern matching and text manipulation in Python. They allow you to search, extract, and manipulate strings based on specific patterns, making them ideal for tasks like data cleaning, text extraction, and validation. With regex, you can efficiently handle tasks like finding email addresses, phone numbers, or specific word patterns in large datasets. They are widely used in text processing and provide a concise way to match complex text patterns flexibly and efficiently. Some resources to learn regular expression can be found here. You can use either any or all resources to learn.
* [Python Tutorial: re Module - How to Write and Match Regular Expressions](https://www.youtube.com/watch?v=K8L6KVGG-7o)
* [Learn Regular Expressions (Regex) - Crash Course for Beginners](https://www.youtube.com/watch?v=ZfQFUJhPqMM)
* [Regular Expressions (Regex) Tutorial: How to Match Any Pattern of Text](https://youtu.be/sa-TUpSx1JA?si=bkAKsYbxPbEKS_ge)

________________________________________________________________________________________________
## Natural Language Tookl Kit (NLTK)
The Natural Language Toolkit (NLTK) is a comprehensive Python library for working with human language data. It offers a range of tools for text processing, including tokenization, parsing, stemming, lemmatization, and stop-words removal. NLTK is widely used for building language models, text classification, and sentiment analysis. It provides easy access to standard datasets, making it a great starting point for anyone interested in natural language processing (NLP) and text analysis.

**Learning Resources:**
* [NLTK: Official Documentation](https://www.nltk.org/): Official Documentation for NLTK - Always go through the documentation to understand and fix your errors, you can always try finding language errors in the documentation if you ever feel bored :)
* [NLTK with Python for NLP](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL): first 3 videos will suffice.
* [NLTK Tutorial](https://www.youtube.com/playlist?list=PLS1QulWo1RIZDws-_0Bfw5FZFmQJWtMl1): The first 4 videos will suffice
________________________________________________________________________________________________
