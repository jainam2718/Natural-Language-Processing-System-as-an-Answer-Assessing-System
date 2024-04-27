import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import random

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()

nltk.download('wordnet')
nltk.download('punkt')

def get_pos_tag(word):
    return nltk.pos_tag([word])[0][1]

def get_synonyms(word, pos_tag):
    synonyms = []
    for syn in wordnet.synsets(word):#, pos=wordnet._wordnet_postag_map.get(pos_tag)):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def replace_with_synonym(word, synonyms):
    if synonyms:
#         return synonyms[0]  # Replace with the first synonym
        return random.choice(synonyms)
    return word  # If no synonyms found, retain the original word

def augment_with_synonyms(text):
    sentences = sent_tokenize(text)
    augmented_sentences = []
    for sentence in sentences:
        tokenized = word_tokenize(sentence)
        augmented_tokens = []
        for token in tokenized:
            pos_tag = get_pos_tag(token)
            synonyms = get_synonyms(token, pos_tag)
            augmented_tokens.append(replace_with_synonym(token, synonyms))
        augmented_sentence = ' '.join(augmented_tokens)
        augmented_sentences.append(augmented_sentence)
    augmented_text = ' '.join(augmented_sentences)
    return augmented_text

def augment_samples(sentences):
    augmented_samples = []
    for senetence in sentences:
        augmented_samples.append(augment_with_synonyms(senetence))
    return augmented_samples