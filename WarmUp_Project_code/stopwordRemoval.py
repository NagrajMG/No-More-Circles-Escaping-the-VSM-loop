from util import *

# Add your import statements here
nltk.download('stopwords')
from nltk.corpus import stopwords



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
		# Function to remove stopwords from a list of tokens
		def stopwords_sieve(tokens , stopwords_list):
			return [token for token in tokens if token.lower() not in stopwords_list]
		# Loop over the list of list of tokens and remove stopwords from each list
		stopwordRemovedText = [stopwords_sieve(tokens , stopwords_list) for tokens in text]
		return stopwordRemovedText
    
	

	


	