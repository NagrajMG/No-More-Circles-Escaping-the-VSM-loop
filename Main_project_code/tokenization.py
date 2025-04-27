from util import *

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			
		"""
		pattern = f"{subword_connectors}|{whitespaces}|{delimiter}"
		# Function to tokenize a sentence
		def tokenize_text(string_sent):
			tokens = re.split(pattern, string_sent)
			return [token for token in tokens if token != ""]

		# Loop over the list of sentences and tokenize each sentence
		tokenizedText = [tokenize_text(string_sent) for string_sent in text]
		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
        # Tokenization using Penn Tree Bank Tokenizer
		tokenizer = TreebankWordTokenizer()
		# Loop over the list of sentences and tokenize each sentence
		tokenizedText = [tokenizer.tokenize(sentence) for sentence in text]
		
		return tokenizedText