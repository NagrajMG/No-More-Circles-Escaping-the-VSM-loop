from util import *

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
        # Splitting the text based on delimiters like periods(.), question mark(?), exclamation mark(!)
		unprocessedSegmentedText = re.split(delimiter, text) 
		# Removing unnecessary whitespace && empty strings if generated from splitting ellipsis(...) etc.
		segmentedText = [sentence.strip() for sentence in unprocessedSegmentedText if sentence.strip() != ''] 
		
		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence

		"""
		# Segmentation using Punkt Tokenizer
		tokenizer = PunktSentenceTokenizer()
		# Segmenting the document body
		segmentedText = tokenizer.tokenize(text)
		return segmentedText