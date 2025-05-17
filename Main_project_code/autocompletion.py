from util import *

# Maximum ASCII value in the corpus
vocab_size= maxASCII() 

# Trie data structure for autocompletion of queries
class TrieNode(): 
    def __init__(self):
        self.children=[None for _ in range(vocab_size)] 

class Trie():
    def __init__(self):
        self.vocab_size=vocab_size
        queries_json = json.load(open("./cranfield/cran_queries.json", 'r'))[:]
        # Loading queries from the JSON file
        self.queries=[doc['query'] for doc in queries_json] 

    def isTerminal(self, node): 
        #check if the node is the terminal node
        for n in node.children:
            if n!=None:
                return False
        return True

    def insertCharacter(self, root, key):
        # Inserting the key in the trie
        currNode=root
        for c in key:
            # if the child node was None, we are creating a new node
            if currNode.children[ord(c)]==None: 
                # Craeting a new node with the ASCII value of the character
                newNode=TrieNode()
                currNode.children[ord(c)]=newNode
            currNode=currNode.children[ord(c)]

            
    def searchPrefix(self, root, pref): 
        # Searching the prefix in the trie
        pref=pref.lower()
        currNode=root
        final="" # New query automatically generated

        for char in pref: 
            idx=ord(char)
            #get the child node belong to that character
            node=currNode.children[idx] 
            #if the child node was not present, return False
            if node==None: 
                return False
            #if the child node was present, add the character to the final query
            final += char 
            currNode= node
        query_list=[]
     
        self.probableQueries(currNode, "", query_list) 

        for i, s in enumerate(query_list):
            query_list[i]= final + s
        return self.printQueries(query_list) 
    # Printing the query list
    def printQueries(self, query_list): 
        if(len(query_list)==1):
            return query_list[0]
        print('Choose any one of the auto-completed query: ') 
        for i, query in enumerate(query_list):
            print(f'{i+1}. {query}')
        idx=int(input())-1
        return query_list[idx]
    # Using DFS to find the probable queries
    def probableQueries(self, currNode, s1, l1): 
        if currNode==None:
            return
        for i in range(self.vocab_size):
            self.probableQueries(currNode.children[i], s1+chr(i), l1)
        if self.isTerminal(currNode):
            l1.append(s1)
    #create a trie data structure
    def makeTS(self): 
        root=TrieNode()
        for query in self.queries:
            self.insertCharacter(root, query)
        return root

    def takeInput(self, query): 
        root=self.makeTS() 
        automated_query=self.searchPrefix(root, query) 
        if(automated_query==False):
            print('No queries present in the trie')
            return query
        else:
            print(f'The autocompleted query using Trie is: {automated_query}')
            return automated_query
