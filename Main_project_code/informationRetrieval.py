from util import *
from collections import defaultdict
import numpy as np
from math import log10, sqrt
class InformationRetrieval():

    def __init__(self):
        self.index = None
        # self.start_time = None
        # self.end_time = None
        self.docs = None
    def buildIndex(self, docs, docIDs):
        # self.start_time = time.time()
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
        self.docs = docs
        posting = {}
        for id in docIDs:
            doc = docs[id-1]
            for sentence in doc:
                for word in sentence: # sub-sub list is a sentence
                    if word.isalpha():
                        word = word.lower()
                        if word not in posting:
                            posting[word] = []
                        # Add the document ID to the posting list for the word
                        posting[word].append(id)    
        self.index = posting
        self.IDFvalues(self.index, self.docs)
        self.transformDocs()

    def IDFvalues(self, index, docs):
            """
            Calcalating the IDF values
            """
            idf = {}
            # Number of documents
            D = len(docs)
            # Iterating over keys of postings
            for key in index.keys():
                d = len(index[key])
                idf[key] = math.log2(D/d)
            self.idf = idf  

    def transformDocs(self):
        """Building the TFIDF for the documents"""
        D=len(self.docs)
        docvectors=[]
        # Iterating over the documents
        for id in range(D):
            # Initializing the document vector
            docvector=np.zeros(len(self.index))
            for keyid , key in enumerate(self.index): 
                if self.docs[id]!=None: 
                    # Calculating the term frequency for the document
                    tf = self.docs[id].count(key) 
                    docvector[keyid]=  tf  *  self.idf.get(key,0)
            docvectors.append(docvector)

        # Storing the document vectors with tfidf 
        self.docvectors=docvectors   
        print(f"Shape of document vectors: ",np.array(docvectors).shape)  

    def transformQuery(self, queries):
        """Building the TFIDF for the query"""
        Q = len(queries)
        queryvectors=[] 
        # Iterating over the queries
        for queryid in range(Q):
            # Initializing the query vector
            queryvector=np.zeros(len(self.index))
            for keyid , key in enumerate(self.index): 
                # Calculating the term frequency for the query
                tf = queries[queryid].count(key) 
                queryvector[keyid]=  tf  *  self.idf.get(key,0)
            queryvectors.append(queryvector)
        # Storing the qyery vectors with tfidf 
        self.queryvectors=queryvectors   
        print(f"Shape of query vectors: ",np.array(queryvectors).shape)        

    def LSI(self, k, docvectors, queryvectors): 
        """
        Apply Latent Semantic Indexing (LSI) to the document and query vectors.
        Parameters
        ----------
        arg1 : int
                The number of latent concepts (dimensions) to retain.
        arg2 : list
                A list of lists where each sub-list is a document and each sub-sub-list is a sentence of the document
        arg3 : list
                A list of lists where each sub-list is a query and each sub-sub-list is a sentence of the query
        Returns
        -------
        tuple
        A tuple containing two NumPy arrays:
        - The first array consists of the hidden concepts document vectors.
        - The second array consists of the hidden concepts query vectors.        
"""     
        # Perform Singular Value Decomposition (SVD) on the document vectors
        U, S, VT = np.linalg.svd(np.array(docvectors).T)

        U_k = U[:, :k] # Retain the first k columns of U
        VT_k = VT[:k, :] # Retain the first k rows of VT
        S_k = np.diag(S[:k]) # Create a diagonal matrix of the first k singular values
        
        hidden_docs = []
        hidden_queries = []

        # Iterating through all queries
        for query in queryvectors:
            # Transformed the query into the latent space using the SVD components
            hidden_query = np.dot(query, U_k)@np.linalg.inv(S_k)
            hidden_queries.append(hidden_query)
        # Iterating through all documents
        for id in range(len(docvectors)):
            # Transformed the document into the latent space using the SVD components
            hidden_doc = VT_k.T[id]   # Here, shape(VT_k) is (concepts, documents) here, transpose for accesing documents
            hidden_docs.append(hidden_doc)

        return np.array(hidden_docs), np.array(hidden_queries)
    
    def rank(self, queries, method):
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
        
    
    
    def orderDocs(self, docvectors, queryvectors, k): 
        """ Order the documents based on their relevance to the queries using cosine similarity"""
        doc_IDs_ordered=[]
		# Iterating over the queries
        for query_num, query in enumerate(queryvectors): 
            temp={}
            # Iterating over the documents
            for doc_id in range(len(docvectors)): 
                #Computing the norm of the query vector
                queryNorm=np.linalg.norm(query) 
                #Computing the norm of the document vector
                docNorm=np.linalg.norm(docvectors[doc_id]) 
                if queryNorm!=0 and docNorm!=0:

                    cosine=np.dot(query, docvectors[doc_id])/(queryNorm*docNorm) 
                else:
                    cosine=0
                temp[doc_id+1] = cosine #Storing the similarity score
            scores = sorted(temp.items(), key=lambda item: item[1], reverse=True)[:k] 
            doc_IDs_ordered.append([doc[0] for doc in scores])
            # self.end_time = time.time()
        return doc_IDs_ordered    