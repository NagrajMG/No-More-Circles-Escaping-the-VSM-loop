from util import *

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
		# Loop over the list of Tokens lists and stem each Token in each list
		reducedText = [[stemmer.stem(token) for token in sentence] for sentence in text]
	    
		return reducedText
	

	def lemmatize(self, text):
        
		"""
		Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""
		# Lemmatization using WordNet Lemmatizer
		lemmatizer=WordNetLemmatizer() 
		reducedText = []
		for sentence in text:
			# Get the POS tags for each token in the sentence
			pos_tags = nltk.pos_tag(sentence)
			# Lemmatize each token in the sentence using its corresponding POS tag
			wordnet_tags = list(map(lambda x: (x[0], pos_mapping(x[1])), pos_tags))
			lemmatized_sentence=[]
			for word, tag in wordnet_tags:
				if tag is not None:
					lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
				else:
					lemmatized_sentence.append(word)
			reducedText.append(lemmatized_sentence)
		
		# Return the lemmatized text
		return reducedText