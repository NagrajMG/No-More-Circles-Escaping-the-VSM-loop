from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from autocompletion import *
from sys import version_info
import argparse
from util import *
from spell_check import *
# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()
		self.informationRetriever = InformationRetrieval()
		self.evaluator = Evaluation()
		self.autocompleter = Trie()
		self.start_time = None
		self.end_time = None
		# change k for models with less retrieved docs
		self.k = 10
	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)
    
	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		if self.args.reducer == "lemmatizer":
			return self.inflectionReducer.lemmatize(text)
		elif self.args.reducer == "stemmer":
			return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		if self.args.stopwords == "corpus_based":
			return self.stopwordRemover.fromCorpus(text)
		elif self.args.stopwords == "nltk":
			return self.stopwordRemover.fromList(text)
		


	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		
		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs


	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""
		self.start_time = time.time()
		if self.args.model == "LSI" or self.args.model == None:
			self.args.out_folder += "final_model/"
		else:
			self.args.out_folder += "baseline/"

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)

		# Accessing the model args
		model_args = self.args.model
        
		# If the users has some new documents, we will perform Recomputation on SVD 
		recompute = (self.args.recompute == "True")
       
		# Rank the documents for each query and the method to use
		doc_IDs_ordered = self.informationRetriever.rank(processedQueries, model_args, recompute = recompute)

		self.end_time = time.time()
		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []

		K = self.k + 1
		print("Using the value of k for evaluation = " + str(K - 1))
		
		for k in range(1, K):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			print("Precision, Recall and F-score @ " +  
				str(k) + " : " + str(precision) + ", " + str(recall) + 
				", " + str(fscore))
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)
			print("MAP, nDCG @ " +  
				str(k) + " : " + str(MAP) + ", " + str(nDCG))
        
		# self.evaluator.saveRecallstem()
		# self.evaluator.savePrecisionstem()
		# self.evaluator.saveFscorestem()
		# self.evaluator.saveAPstem()
		# self.evaluator.savenDCGstem()

		# Plot the metrics and save plot 
		plt.plot(range(1, K), precisions, label="Precision")
		plt.plot(range(1, K), recalls, label="Recall")
		plt.plot(range(1, K), fscores, label="F-Score")
		plt.plot(range(1, K), MAPs, label="MAP")
		plt.plot(range(1, K), nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		plt.savefig(args.out_folder + "eval_plot.png")
    
	def tuningLSI(self):
		
		if self.args.model == "LSI" :
			self.args.out_folder += "final_model/"
		else:
			self.args.out_folder += "baseline/"

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)

		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

		k_values = list(range(10, 251, 10))
		recall_at_k = 6
		best_k_recall = -1
		best_recall = -1
		recall_scores = []
		MAP_scores = []
		best_k_map = -1
		best_map = -1

        # Transform the documents into tfidf vectors
		queryvectors = self.informationRetriever.transformQuery(processedQueries)

		# Preparing a log file to store the results
		log_path = self.args.out_folder + "lsi_tuning_log.csv"

		with open(log_path, mode='w', newline='') as logfile:
			writer = csv.writer(logfile)
			writer.writerow(["k", "Recall@6", "MAP@6"])
			for k in k_values:
				print(f"Evaluating k = {k}...")
				docs_k, queries_k = self.informationRetriever.LSI(k, self.informationRetriever.docvectors, queryvectors, recompute = True)
				ranked_docs = self.informationRetriever.orderDocs(docs_k, queries_k, k=recall_at_k)
				recall = self.evaluator.meanRecall(ranked_docs, query_ids, qrels, recall_at_k)
				recall_scores.append(recall)
				MAP = self.evaluator.meanAveragePrecision(ranked_docs, query_ids, qrels, recall_at_k)
				MAP_scores.append(MAP)
				print(f"Recall@{recall_at_k} = {recall:.4f} and MAP@{recall_at_k} = {MAP:.4f}")
				# Write the k and recall to the log file
				writer.writerow([k, recall, MAP])
			
				if recall > best_recall:
					best_k_recall = k
					best_recall = recall
				if MAP > best_map:
					best_k_map = k
					best_map = MAP
		
		plt.figure(figsize=(10, 5))
		plt.plot(k_values, recall_scores, label=f"Recall@{recall_at_k}", marker='o')
		plt.plot(k_values, MAP_scores, label=f"MAP@{recall_at_k}", marker='s')
		plt.xlabel("SVD Components (k)")
		plt.ylabel("Score")
		plt.title("LSI Hyperparameter Tuning - Recall and MAP vs k")
		plt.grid(True)
		plt.legend()
		plt.savefig(self.args.out_folder + "lsi_tuning_plot.png")
		plt.show()

		print(f"\nBest SVD k = {best_k_recall} with Recall@{recall_at_k} = {best_recall:.4f}")
		print(f"Best SVD k = {best_k_map} with MAP@{recall_at_k} = {best_map:.4f}")
		# Plot the recall and MAP scores
		print(f"Log saved at: {log_path}")		
		



	def handleCustomQuery(self):
		
		if self.args.model == "LSI" :
			self.args.out_folder += "final_model/"
		else:
			self.args.out_folder += "baseline/"

		#Get query
		print("Enter query below")
		query = input()
		
		# If the user wants to use autocomplete, we will use the Trie data structure to get the query
		if(self.args.autocomplete):
			query = self.autocompleter.takeInput(query)
			
		self.start_time = time.time()
	
		# Process documents
		processedQuery = self.preprocessQueries([query])[0]

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for the query
		model_args = self.args.model
		# If the users has some new documents, we will perform Recomputation on SVD 
		recompute = (self.args.recompute == "True")

		doc_IDs_ordered = self.informationRetriever.rank([processedQuery], method = model_args, recompute = recompute)[0]
        
		self.end_time = time.time()

		# Print the IDs of first five documents
		print("\nTop five document IDs : ")
		for id_ in doc_IDs_ordered[:10]:
			print(id_)



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')
    
	# Tunable parameters as external arguments
	parser.add_argument('-model', default='LSI', help="Model Type [VSM|LSI|clustering]")
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "Main_project_output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-stopwords', default = "nltk",
	                    help = "Stopword Removal Type [nltk|corpus_based]")
	parser.add_argument('-reducer', default = "lemmatizer", 
					    help = "Inflection Reduction Type [stemmer|lemmatizer]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	parser.add_argument('-autocomplete', action='store_true',
                        help="Autocomplete")
	parser.add_argument('-findK', default = "False",
						help = "Find the best k for LSI")
	parser.add_argument('-recompute', default = "False",
						help = "Recompute the SVD for LSI [True|False]")
	
	# Parse the input arguments
	args = parser.parse_args()
    
	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		# start_time = time.time()
		searchEngine.handleCustomQuery()
		# end_time = time.time()
		print("-------------------------------------------------------------------------------------------------------------")
		print(f"Time taken by the IR system to output ten relevant documents: " + str(searchEngine.end_time - searchEngine.start_time) + " seconds")
		print("-------------------------------------------------------------------------------------------------------------")
	elif args.findK == "True" and args.model == "LSI":
		searchEngine.tuningLSI()
	else:
		searchEngine.evaluateDataset()
		print("-------------------------------------------------------------------------------------------------------------")
		print(f"Time taken by the IR system to evaluate the " + str(searchEngine.args.model)+ " model: " + str(searchEngine.end_time - searchEngine.start_time) + " seconds")
		print("-------------------------------------------------------------------------------------------------------------")
