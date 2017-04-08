# nlp-file-organizer
Uses NLTK and SpaCy to group files into folders based on similar word meaning in the titles.

## Necessary libraries

In order to run the program you'll need to install:
 * Numpy
 * Scipy
 * NLTK
 * SpaCy

Once you install those through pip, you'll need to install the Corpora for NLTK and SpaCY.

```python
import nltk

nltk.download()
```

```bash
python3 -m spacy download en
```

For more accurate word recognition, you can [download](https://github.com/explosion/spacy-models/blob/master/README.md) the large SpaCy corpus (en\_core\_web\_md):

```bash
pip3 install en_core_web_md-X.X.X.tar.gz
python3 -m spacy link en_core_web_md en
```

Once the dependencies are installed, run the program with `python3 nlp-file-organizer.py DIRECTORY\_TO\_ORGANIZE MAX_FOLDERS`.