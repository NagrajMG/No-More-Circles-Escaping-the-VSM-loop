from util import *
from math import log2
# Add your import statements here

class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""
        # Number of retrieved documents
		num_docs = len(query_doc_IDs_ordered)
		# Error handling if k > num_docs
		if k > num_docs:
			print("Error: k cannot be greater than the number of retrieved documents.")
			return -1
		# Number of relevant documents in the top k retrieved documents
		relevant_docs = 0
		# Iterate through the top k retrieved documents
		for id in query_doc_IDs_ordered[:k]:
			# Check if the document ID is in the list of relevant documents
			if id in true_doc_IDs:
				relevant_docs += 1
        # After iterating through the top k retrieved documents, calculate precision
		precision = relevant_docs / k if k > 0 else 0

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		self.precision_vsm = []
		self.precision_lsi = []
        # Number of queries
		num_queries = len(query_ids)
		# Error handling if num_queries is 0
		if num_queries == 0:
			print("Error: No queries provided.")
			return -1
		# Initialize precision list 
		precisions = []
		# Iterate through each query
		for i in range(num_queries):
			# Get the query ID and the list of relevant documents
			query_id = query_ids[i]
			query_docs = doc_IDs_ordered[i]
			# Get the list of relevant documents for the query
			relevant_docs = [int(doc['id']) for doc in qrels if int(doc['query_num']) == query_id]
            # Calculate precision for the current query
			precision = self.queryPrecision(query_docs, query_id, relevant_docs, k)
			# Append the precision to the list
			precisions.append(precision)
			# self.precision_vsm.append(precision)
			self.precision_lsi.append(precision)
		# Calculate mean precision
		meanPrecision = sum(precisions) / num_queries 
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		# Number of retrieved documents
		num_docs = len(query_doc_IDs_ordered)
		# Number of relevant documents
		num_relevant_docs = len(true_doc_IDs)
		# Error handling if k > num_docs
		if k > num_docs:
			print("Error: k cannot be greater than the number of retrieved documents.")
			return -1
		# Number of relevant documents in the top k retrieved documents
		relevant_docs = 0
		# Iterate through the top k retrieved documents
		for id in query_doc_IDs_ordered[:k]:
			# Check if the document ID is in the list of relevant documents
			if id in true_doc_IDs:
				# Increment the count of relevant documents
				relevant_docs += 1
		# After iterating through the top k retrieved documents, calculate recall
		recall = relevant_docs / num_relevant_docs if num_relevant_docs > 0 else 0
		return recall

	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""
		self.recall_vsm = []
		self.recall_lsi = []
		# Number of queries
		num_queries = len(query_ids)
		# Error handling if num_queries is 0
		if num_queries == 0:
			print("Error: No queries provided.")
			return -1
		# Initialize recall list
		recalls = []
		# Iterate through each query
		for i in range(num_queries):
			# Get the query ID and the list of relevent documents
			query_id = int(query_ids[i])
			query_docs = doc_IDs_ordered[i]
			# Get the list of relevant documents for the query
			relevant_docs = [int(doc['id']) for doc in qrels if int(doc['query_num']) == query_id]
			# Calculate recall for the current query=
			recall = self.queryRecall(query_docs, query_id, relevant_docs, k)
			# Append the recall to the list
			recalls.append(recall)
			# self.recall_vsm.append(recall)
			self.recall_lsi.append(recall)
		# Calculate mean recall
		meanRecall = sum(recalls) / num_queries 
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""
        # Calculate precision and recall
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		beta = 0.5
		# Calculate fscore
		if precision + recall == 0:
			fscore = 0
		else:
			fscore = ((1 + beta**2) * precision * recall) / (beta ** 2 * precision + recall)
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""
		# Number of queries
		num_queries = len(query_ids)
		# Error handling if num_queries is 0
		self.fscore_vsm = []
		self.fscore_lsi = []
		if num_queries == 0:
			print("Error: No queries provided.")
			return -1
		# Initialize fscore list
		fscores = []
		# Iterate through each query
		for i in range(num_queries):
			# Get the query ID and the list of retrieved documents
			query_id = int(query_ids[i])
			query_docs = doc_IDs_ordered[i]
			# Get the list of relevant documents for the query
			relevant_docs = [int(doc['id']) for doc in qrels  if int(doc['query_num']) == query_id]
			# Calculate fscore for the current query
			fscore = self.queryFscore(query_docs, query_id, relevant_docs, k)
			# Append the fscore to the list
			fscores.append(fscore)
			# self.fscore_vsm.append(fscore)
			self.fscore_lsi.append(fscore)
		# Calculate mean fscore
		meanFscore = sum(fscores) / num_queries
		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""
		# Create a dictionary for gathering and storing key-value pairs
		relevance_for_ids = {}
		# Getting relevance scores for the documents as per the query ID
		for doc in true_doc_IDs:
			if (int(doc['query_num']) == query_id):
				# Calcualting relevance score
				maximum_position = 4
				# Handling zeros in the relevance score
				maximum_position += 1
				# Feeding the relevance score 
				relevance_score = maximum_position - int(doc['position'])
				# Creating a dictionary of document IDs and their relevence score
				relevance_for_ids[int(doc['id'])] = relevance_score
        
		# Initialize DCG@k
		DCG_k = 0.0
		iterations = min(k, len(query_doc_IDs_ordered))
		# In the list of retrieved documents, matching ID with previously created dictionary and if found calculating DCG@K
		for i in range(iterations):
			# Get the document ID
			doc_id = query_doc_IDs_ordered[i]
			if doc_id in relevance_for_ids:
				relevance_score = relevance_for_ids[doc_id]
				# Calculate DCG@k
				rank = i + 1
				DCG_k += (((2**relevance_score) - 1 ) / log2(rank + 1))

		# Initialize IDCG@k 
		sorted_scores = sorted(relevance_for_ids.values(), reverse = True)
		# Calculate IDCG@k
		rank1 = 1
		j = 0
		IDCG_k = 0.0
		while j < len(sorted_scores) and rank1 <= k:
			relevance_score = sorted_scores[j]
			# Calculate IDCG@k
			IDCG_k += (((2**relevance_score) - 1 ) / log2(rank1 + 1))
			j += 1
			rank1 += 1
		# Calculate nDCG@k
		if IDCG_k == 0:
			nDCG = 0.0
		else:
			nDCG = DCG_k / IDCG_k
		# Return nDCG value

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""
		self.ndcg_vsm = []
		self.ndcg_lsi = []
        # Number of queries
		num_queries = len(query_ids)
		# Error handling if num_queries is 0
		if num_queries == 0:
			print("Error: No queries provided.")
			return -1
		# Initialize nDCG list
		nDCGs = []
		# Iterate through each query
		for i in range(num_queries):
			# Get the query ID and the list of retrieved documents
			query_id = int(query_ids[i])
			query_docs = doc_IDs_ordered[i]
			# Fetching nDCG for the current query
			nDCG = self.queryNDCG(query_docs, query_id, qrels, k)
            # Append the nDCG to the list
			nDCGs.append(nDCG)
			# self.ndcg_vsm.append(nDCG)
			self.ndcg_lsi.append(nDCG)
		# Calculate mean nDCG
		meanNDCG = sum(nDCGs) / num_queries

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""
		# Number of retrieved documents
		num_docs = len(query_doc_IDs_ordered)
		# Number of relevant documents
		num_relevant_docs = len(true_doc_IDs)
		# Error handling if k > num_docs
		if k > num_docs:
			print("Error: k cannot be greater than the number of retrieved documents.")
			return -1
		boolean_relevance = [ 1 if ID in true_doc_IDs else 0 for ID in query_doc_IDs_ordered[:k]]
  
		precisions = [self. queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i + 1) for i in range(k)]
		# Calculate average precision
		precision_at_k = [precisions[i] * boolean_relevance[i] for i in range(k)]
		avgPrecision = sum(precision_at_k) / num_relevant_docs if num_relevant_docs > 0 else 0
		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""
		self.ap_vsm = []
		self.ap_lsi = []
        # Number of queries
		num_queries = len(query_ids)
		# Error handling if num_queries is 0
		if num_queries == 0:
			print("Error: No queries provided.")
			return -1
		# Initialize average precision list
		avg_precisions = []
		# Iterate through each query
		for i in range(num_queries):
			# Get the query ID and the list of retrieved documents
			query_id = int(query_ids[i])
			query_docs = doc_IDs_ordered[i]
			# Get the lsit of relevant documents for the query
			relevant_docs = [int(doc['id']) for doc in q_rels if int(doc['query_num']) == query_id]
			# Calculate average precision for the current query
			avg_precision = self.queryAveragePrecision(query_docs, query_id, relevant_docs, k)
			# Append the average precision to the list
			avg_precisions.append(avg_precision)
			# self.ap_vsm.append(avg_precision)
			self.ap_lsi.append(avg_precision)

		# Calculate mean average precision
		meanAveragePrecision = sum(avg_precisions) / num_queries if num_queries > 0 else 0
		
		return meanAveragePrecision
	

# Save the evaluation scores to CSV files (in the TestScores directory)	
	def saveRecallLSI(self, filename="TestScores/recallLSI.csv"):
		with open(filename, "w") as f:
			f.write("query_id,recall\n")
			for i, score in enumerate(self.recall_lsi, 1):
				f.write(f"{i},{score:.4f}\n")

	
	
	def saveRecallVSM(self, filename="TestScores/recallVSM.csv"):
		with open(filename, "w") as f:
			f.write("query_id,recall\n")
			for i, score in enumerate(self.recall_vsm, 1):
				f.write(f"{i},{score:.4f}\n")
	
	def savePrecisionVSM(self, filename="TestScores/precisionVSM.csv"):
		with open(filename, "w") as f:
			f.write("query_id,precision\n")
			for i, score in enumerate(self.precision_vsm, 1):
				f.write(f"{i},{score:.4f}\n")

	def saveFscoreVSM(self, filename="TestScores/fscoreVSM.csv"):
		with open(filename, "w") as f:
			f.write("query_id,fscore\n")
			for i, score in enumerate(self.fscore_vsm, 1):
				f.write(f"{i},{score:.4f}\n")


	def savenDCGVSM(self, filename="TestScores/nDCGVSM.csv"):
		with open(filename, "w") as f:
			f.write("query_id,nDCG\n")
			for i, score in enumerate(self.ndcg_vsm, 1):
				f.write(f"{i},{score:.4f}\n")

	def saveAPVSM(self, filename="TestScores/APVSM.csv"):
		with open(filename, "w") as f:
			f.write("query_id,AP\n")
			for i, score in enumerate(self.ap_vsm, 1):
				f.write(f"{i},{score:.4f}\n")
	
	
	
	def savePrecisionLSI(self, filename="TestScores/precisionLSI.csv"):
		with open(filename, "w") as f:
			f.write("query_id,precision\n")
			for i, score in enumerate(self.precision_lsi, 1):
				f.write(f"{i},{score:.4f}\n")

	def saveFscoreLSI(self, filename="TestScores/fscoreLSI.csv"):
		with open(filename, "w") as f:
			f.write("query_id,fscore\n")
			for i, score in enumerate(self.fscore_lsi, 1):
				f.write(f"{i},{score:.4f}\n")


	def savenDCGLSI(self, filename="TestScores/nDCGLSI.csv"):
		with open(filename, "w") as f:
			f.write("query_id,nDCG\n")
			for i, score in enumerate(self.ndcg_lsi, 1):
				f.write(f"{i},{score:.4f}\n")

	def saveAPLSI(self, filename="TestScores/APLSI.csv"):
		with open(filename, "w") as f:
			f.write("query_id,AP\n")
			for i, score in enumerate(self.ap_lsi, 1):
				f.write(f"{i},{score:.4f}\n")



	def savePrecisioncb(self, filename="TestScores/Stopwords/precisionCB.csv"):
		with open(filename, "w") as f:
			f.write("query_id,precision\n")
			for i, score in enumerate(self.precision_lsi, 1):
				f.write(f"{i},{score:.4f}\n")

	def saveFscorecb(self, filename="TestScores/Stopwords/fscoreCB.csv"):
		with open(filename, "w") as f:
			f.write("query_id,fscore\n")
			for i, score in enumerate(self.fscore_lsi, 1):
				f.write(f"{i},{score:.4f}\n")


	def savenDCGcb(self, filename="TestScores/Stopwords/nDCGCB.csv"):
		with open(filename, "w") as f:
			f.write("query_id,nDCG\n")
			for i, score in enumerate(self.ndcg_lsi, 1):
				f.write(f"{i},{score:.4f}\n")

	def saveAPcb(self, filename="TestScores/Stopwords/APCB.csv"):
		with open(filename, "w") as f:
			f.write("query_id,AP\n")
			for i, score in enumerate(self.ap_lsi, 1):
				f.write(f"{i},{score:.4f}\n")

	def saveRecallcb(self, filename="TestScores/Stopwords/RecallCB.csv"):
		with open(filename, "w") as f:
			f.write("query_id,recall\n")
			for i, score in enumerate(self.recall_lsi, 1):
				f.write(f"{i},{score:.4f}\n")


	def savePrecisionstem(self, filename="TestScores/Reducer/precisionstem.csv"):
		with open(filename, "w") as f:
			f.write("query_id,precision\n")
			for i, score in enumerate(self.precision_lsi, 1):
				f.write(f"{i},{score:.4f}\n")

	def saveFscorestem(self, filename="TestScores/Reducer/fscorestem.csv"):
		with open(filename, "w") as f:
			f.write("query_id,fscore\n")
			for i, score in enumerate(self.fscore_lsi, 1):
				f.write(f"{i},{score:.4f}\n")


	def savenDCGstem(self, filename="TestScores/Reducer/nDCGstem.csv"):
		with open(filename, "w") as f:
			f.write("query_id,nDCG\n")
			for i, score in enumerate(self.ndcg_lsi, 1):
				f.write(f"{i},{score:.4f}\n")

	def saveAPstem(self, filename="TestScores/Reducer/APstem.csv"):
		with open(filename, "w") as f:
			f.write("query_id,AP\n")
			for i, score in enumerate(self.ap_lsi, 1):
				f.write(f"{i},{score:.4f}\n")

	def saveRecallstem(self, filename="TestScores/Reducer/Recallstem.csv"):
		with open(filename, "w") as f:
			f.write("query_id,recall\n")
			for i, score in enumerate(self.recall_lsi, 1):
				f.write(f"{i},{score:.4f}\n")