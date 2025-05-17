from util import *

# Add your import statements here
from nltk.stem import PorterStemmer

class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""
		# Stemming using Porter Stemmer
		stemmer = PorterStemmer()
		# Loop over the list of Tokens list and stem each Token in each list
		reducedText = [[stemmer.stem(token) for token in sentence] for sentence in text]
		return reducedText


