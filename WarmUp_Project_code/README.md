### Assignment 1 ###

This folder contains the template code for a search engine application. 

main.py - The main module that contains the outline of the Search Engine. Do not change anything in this file.
util.py - An extra file where you can add any additional processing or utility functions that you may need for any of the sub-tasks.
sentenceSegmentation.py, tokenization.py, inflectionReduction.py and stopwordRemoval.py - Implement the corresponding sub-tasks inside the functions in these files.

More files corresponding to each sub-task will be provided as the assignment progresses, along with updated versions of main.py

To test your code, run main.py with the appropriate arguments
Usage: main.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)] 
When the -custom flag is passed, the system will take a query from the user as input. When the flag is not passed, all the queries in the Cranfield dataset are considered, for example:
> python main.py -custom
> Enter query below
> Papers on Aerodynamics
This will generate *queries.txt files in the OUTPUT FOLDER after each stage of preprocessing of the query and *docs.txt files in the OUTPUT FOLDER after each stage of preprocessing of the documents.


### Assignment 2 ###

This folder contains the additional files required for Part 2 of the assignment, involving building a search engine application. Note that this code works for both Python 2 and Python 3. 

The following files have been added:
informationRetrieval.py and evaluation.py - Implement the corresponding tasks inside the functions in these files.

The following file has been updated:
main.py - The main module that contains the outline of the Search Engine. It has been updated to include calls to the information retrieval and evaluation tasks, in addition to the tasks solved in Part 1 of the assignment. Do not change anything in this file.

For this part of the assignment, you are advised to make a copy of the completed code from Part 1 of the assignment - replace the main file with the updated version and add and fill in the new files (informationRetrieval.py and evaluation.py).

To test your code, run main.py as before with the appropriate arguments.
Usage: main.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)] 

When the -custom flag is passed, the system will take a query from the user as input. For example:
> python main.py -custom
> Enter query below
> Papers on Aerodynamics
This will print the IDs of the five most relevant documents to the query to standard output.

When the flag is not passed, all the queries in the Cranfield dataset are considered and precision@k, recall@k, f-score@k, nDCG@k and the Mean Average Precision are computed.

In both the cases, *queries.txt files and *docs.txt files will be generated in the OUTPUT FOLDER after each stage of preprocessing of the documents and queries.