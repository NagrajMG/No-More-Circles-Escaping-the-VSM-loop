from util import *
from collections import defaultdict
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
        self.docIDs = docIDs
        posting = {}
        # Iterating over the documents
        for id in docIDs: 
            doc=docs[id-1]
            #Iterating over the words in the document
            for word in doc: 
                 #Checking if the word is an alphabet
                 if word.isalpha(): 
                    #Checking if the word is present in the posting dictionary
                    if word not in posting: 
                        posting[word]=[]
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
                idf[key] = log10(D/d)
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
        self.docvectors = docvectors  
        # np.save("Main_project_code/fast_search/output/originalDocs.npy", docvectors)
        # print(f"Shape of document vectors: ",np.array(docvectors).shape)  

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
          
        # print(f"Shape of query vectors: ",np.array(queryvectors).shape) 
        return queryvectors    

    def LSI(self, k, docvectors, queryvectors, recompute): 
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
        hidden_docs = None
        U_k = None
        S_k = None
        if recompute:
            # Perform Singular Value Decomposition (SVD) on the document vectors
            U, S, VT = np.linalg.svd(np.array(docvectors).T)
            U_k = U[:, :k] # Retain the first k columns of U
            VT_k = VT[:k, :] # Retain the first k rows of VT
            S_k = np.diag(S[:k]) # Create a diagonal matrix of the first k singular values
            hidden_docs = (S_k @ VT_k).T
            # Save the hidden documents and SVD components to files
            np.save("Main_project_code/fast_search/output/approximatedDocs.npy", hidden_docs)
            np.save("Main_project_code/fast_search/output/Hanger.npy", U_k)
            np.save("Main_project_code/fast_search/output/Stretcher.npy", S_k)
            print("SVD computed and saved.")
        else:
            hidden_docs = np.load("Main_project_code/fast_search/output/approximatedDocs.npy")
            U_k = np.load("Main_project_code/fast_search/output/Hanger.npy")
            S_k = np.load("Main_project_code/fast_search/output/Stretcher.npy")
            print("Using precomputed SVD.")

        hidden_queries = np.array(queryvectors) @ U_k @ np.linalg.pinv(S_k) 
        return hidden_docs, hidden_queries
    
    def doClustering(self,docvectors,queryvectors):
        """ Perform KMeans clustering on the document vectors."""

        doc_labels = np.load("Main_project_code/fast_search/output/cluster_labels.npy")
        centroids = np.load("Main_project_code/fast_search/output/centroids.npy")

        doc_IDs_ordered = []
        for query in queryvectors:
            queryNorm = np.linalg.norm(query)
            if queryNorm == 0:
                doc_IDs_ordered.append([])
                continue

            # Calculate the cosine similarity between the query and each centroid
            cluster_sim = cosine_similarity(query.reshape(1, -1), centroids)
            best_cluster = np.argmax(cluster_sim)
            # Get the document IDs for the best cluster
            doc_ids_in_cluster = np.where(doc_labels == best_cluster)[0]
            #Compute cosine similarity with only those docs
            temp = {}
            for doc_id in doc_ids_in_cluster:
                doc = docvectors[doc_id]
                docNorm = np.linalg.norm(doc)
                if docNorm != 0:
                    cosine = np.dot(query, doc) / (queryNorm * docNorm)
                else:
                    cosine = 0
                temp[doc_id + 1] = cosine

            scores = sorted(temp.items(), key=lambda item: item[1], reverse=True)[:10]
            doc_IDs_ordered.append([doc[0] for doc in scores])
        return doc_IDs_ordered

    def rank(self, queries, method, recompute):
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
        queryvectors = self.transformQuery(queries) # Transform the queries into tfidf vectors
        # Accessing the document vectors
        docvectors = self.docvectors
        if method=='LSI':
            docvectors_lsi, queryvectors_lsi = self.LSI(k= 160 , docvectors=docvectors, queryvectors=queryvectors, recompute = recompute)
            return self.orderDocs(docvectors_lsi, queryvectors_lsi)
        elif method == 'VSM':
            return self.orderDocs(docvectors, queryvectors)
        elif method == "clustering":
            return self.doClustering(docvectors,queryvectors)
        else:
            print(f'Invalid model argument: {method}')
            print(f'Available Methods: VSM, LSI, clustering')
            quit()

    
    
    def orderDocs(self, docvectors, queryvectors, k = 10): 
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