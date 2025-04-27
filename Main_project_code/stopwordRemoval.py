from util import *

class StopwordRemoval():
	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
        # Curated stopwords list from NLTK
		stopwords_list = set(stopwords.words("english"))
		# Loop over the list of list of tokens and remove stopwords from each list
		stopwordRemovedText = [stopwords_sieve(tokens , stopwords_list) for tokens in text]
		return stopwordRemovedText
	
	def fromCorpus(self, text):
		"""
		Custom stopword removal using a corpus-based approach
		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		# Load stopwords from a file

		with open(r'Main_project_code\corpus_based_stopwords.json', 'r') as f:
			stopwords_list = json.load(f)
		# Loop over the list of list of tokens and remove stopwords from each list
		stopwordRemovedText = [stopwords_sieve(tokens , stopwords_list) for tokens in text]
		return stopwordRemovedText
	
        

	