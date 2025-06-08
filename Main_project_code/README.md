# Search Engine - based upon Cranfield dataset

This project implements a modular and extensible Information Retrieval (IR) system capable of indexing and ranking documents using multiple models, preprocessing strategies, and evaluation metrics.

## Follow the steps to explore the project:
```python
cd No-More-Circles-Escaping-the-VSM-loop
```

Try:
```python
python Main_project_code\main.py 
```
options:
-model MODEL          Model Type [VSM|LSI|clustering]
-dataset DATASET      Path to the dataset folder
-out_folder OUT_FOLDER    Path to output folder
-segmenter SEGMENTER  Sentence Segmenter Type [naive|punkt]
-tokenizer TOKENIZER  Tokenizer Type [naive|ptb]
-stopwords STOPWORDS  Stopword Removal Type [nltk|corpus_based]
-reducer REDUCER      Inflection Reduction Type [stemmer|lemmatizer]
-custom               Take custom query as input
-autocomplete         Autocomplete
-findK FINDK          Find the best k for LSI
-recompute RECOMPUTE  Recompute the SVD for LSI [True|False]

##  Project Directory Structure

```python
├── cranfield/                        # Dataset folder
├── Main_project_code/               # Core project code
│   ├── __pycache__/                 # Compiled Python files (auto-generated)
│   ├── fast_search/                 # Fast search module
│   │   ├── output/                  # Output files for fast search
│   │   └── kMeans.py                # Clustering-based approach
│   ├── autocompletion.py            # Autocompletion logic
│   ├── corpus_based_stopwords.json  # JSON of custom stopwords
│   ├── corpusBasedStopwords.ipynb   # Notebook to generate stopwords
│   ├── evaluation.py                # Evaluation metrics
│   ├── hypothesis.ipynb             # Hypothesis testing
│   ├── inflectionReduction.py       # Inflection reduction logic
│   ├── informationRetrieval.py      # Core IR functionality
│   ├── main.py                      # Main driver script
│   ├── README.md                    # Project report in Markdown
│   ├── sentenceSegmentation.py      # Sentence segmentation logic
│   ├── spell_check.py               # Spell checker module
│   ├── stopwordRemoval.py           # Stopword removal logic
│   ├── tokenization.py              # Tokenization logic
│   ├── util.py                      # Utility functions
│   └── wordnet.py                   # WordNet-based enhancements
├── Main_project_output/             # Output folder
│   ├── baseline/                    # Output from baseline system
│   └── final_model/                 # Output from final model
└── TestScores/                      # Evaluation and test results
    ├── General/                     # General performance results
    ├── Reducer/                     # Results from reducer tests
    └── Stopwords/                   # Impact of stopwords
```
