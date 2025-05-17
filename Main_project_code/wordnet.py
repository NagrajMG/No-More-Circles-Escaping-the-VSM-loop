from util import *

def Synsets(word):
    word_synsets = wordnet.synsets(word)
    print(f"Synsets for '{word}': {word_synsets}")
    return word_synsets

def definitions(word):
    word_synsets = Synsets(word)
    for synset in word_synsets:
        print(synset.name(), "-", synset.definition())

def path_based_similarity(word1, word2):
    max_similarity = 0
    word1_synsets = Synsets(word1)
    word2_synsets = Synsets(word2)
    for word1_synset in word1_synsets:
        for word2_synset in word2_synsets:
            similarity = word1_synset.path_similarity(word2_synset)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
    print(f"Path-based similarity between '{word1}' and '{word2}':", max_similarity)

# # Print synsets and definitions for 'progress' and 'advance'
# Synsets('progress')
# Synsets('advance')
# print("="*50)
# definitions('progress')
# definitions('advance')
# print("="*50)
# # Print path-based similarity between 'progress' and 'advance'
# path_based_similarity('progress', 'advance')
