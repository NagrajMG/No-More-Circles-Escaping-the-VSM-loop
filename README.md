# No-More-Circles-Escaping-the-VSM-loop
This includes code for warm-up project and main project for the course CS6370 Natural Language Processing @ IIT Madras.
- Main project folder has further details regarding the project.
- Report includes tried implementations and experiments.

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
