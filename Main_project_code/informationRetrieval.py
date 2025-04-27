from util import *
from collections import defaultdict
import numpy as np
from math import log10, sqrt
class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.start_time = None
        self.end_time = None
    def buildIndex(self, docs, docIDs):
        self.start_time = time.time()
        """
        Builds the document index in terms of the document IDs and stores it in the 'index' class variable

        Parameters
        ----------
        docs : list
            A list of lists of lists where each sub-list is a document and each sub-sub-list is a sentence of the document
        docIDs : list
            A list of integers denoting IDs of the documents

        Returns
        -------
        None
        """
        # Initialize 'posting' dictionary where:
        # - keys are unique words across all documents.
        # - values are empty lists to later hold [docID, frequency] pairs.
        posting = {tokens: [] for d in docs for sentence in d for tokens in sentence}
        # Adding values to the 'posting' dictionary.
        for i in range(len(docs)):
            # Flatten the document into a single list of tokens for counting term frequency.
            doc = [token for sent in docs[i] for token in sent]
            for sent in docs[i]:
                for token in sent:
                    if token in posting.keys():
                        # Append the [docID, frequency] pair to the posting list for the token.
                        if [docIDs[i], doc.count(token)] not in posting[token]:
                            posting[token].append([docIDs[i], doc.count(token)])
        # Store the postings, document count, and document IDs in the index.                   
        self.index = (posting, len(docs), docIDs)

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        queries : list
            A list of lists of lists where each sub-list is a query and each sub-sub-list is a sentence of the query

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """
        doc_IDs_ordered = []
        # Unpack the indexing built in the buildIndex method
        index, doc_num, doc_ID = self.index

        # Term-document matrix D with shape (doc_num, number of unique terms) intialized to zero
        D = np.zeros((doc_num, len(index.keys())))
        key = list(index.keys())
        for i in range(len(key)):
            for doc in index[key[i]]:
                # Fill the term-document matrix with term frequencies, For 0 based indexing on doc_IDs
                D[doc[0] - 1, i] = doc[1]

        # Computing the IDF values 
        idf = np.zeros((len(key), 1))
        for i in range(len(key)):
            idf[i] = log10(doc_num / len(index[key[i]]))

        # Building TF-IDF matrix
        D = D * idf.T

        # Iterate through each query and compute the cosine similarity with the documents
        for i in range(len(queries)):
            query = defaultdict(list)
            for sent in queries[i]:
                for token in sent:
                    if token in index.keys():
                        query[token] = index[token]
            query = dict(query)
            Q = np.zeros((1, len(key)))
            for token in range(len(key)):
                if key[token] in query.keys():
                    # Binary value for the term in the query showing presence
                    Q[0, token] = 1
            # IDF for stopwords resilience       
            Q = Q * idf.T

            # Cosine similarity between query and documents
            cosines = []
            # Total docs interations
            for doc in range(D.shape[0]):
                similarity = np.dot(Q[0, :], D[doc, :]) / ((np.linalg.norm(Q[0, :]) + 1e-10) * (np.linalg.norm(D[doc, :]) + 1e-10))
                cosines.append(similarity)

            # Sorting the documents based on the cosine similarity scores in descending order
            # Since doc_ID and cosines are parallel lists, we sort them together
            doc_IDs_ordered.append([x for _, x in sorted(zip(cosines, doc_ID), reverse=True)])
            self.end_time = time.time()
            # print("Time taken to rank documents for query is: " + str(self.end_time - self.start_time) + " seconds")
        return doc_IDs_ordered