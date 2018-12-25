
# Self Governing Neural Networks (SGNN): the Projection Layer

> A SGNN's word projections preprocessing pipeline in scikit-learn

In this notebook, we'll use T=80 random hashing projection functions, each of dimensionnality d=14, for a total of 1120 features per projected word in the projection function P. 

Next, we'll need feedforward neural network (dense) layers on top of that (as in the paper) to re-encode the projection into something better. This is not done in the current notebook and is left to you to implement in your own neural network to train the dense layers jointly with a learning objective. The SGNN projection created hereby is therefore only a preprocessing on the text to project words into the hashing space, which becomes spase 1120-dimensional word features created dynamically hereby. Only the CountVectorizer needs to be fitted, as it is a char n-gram term frequency prior to the hasher. This one could be computed dynamically too without any fit, as it would be possible to use the [power set](https://en.wikipedia.org/wiki/Power_set) of the possible n-grams as sparse indices computed on the fly as (indices, count_value) tuples, too.


```python
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.random_projection import SparseRandomProjection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter
from pprint import pprint
```

## Preparing dummy data for demonstration:


```python
class SentenceTokenizer(BaseEstimator, TransformerMixin):
    # char lengths:
    MINIMUM_SENTENCE_LENGTH = 10
    MAXIMUM_SENTENCE_LENGTH = 200
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self._split(X)
    
    def _split(self, string_):
        splitted_string = []
        
        sep = chr(29)  # special separator character to split sentences or phrases.
        string_ = string_.strip().replace(".", "." + sep).replace("?", "?" + sep).replace("!", "!" + sep).replace(";", ";" + sep).replace("\n", "\n" + sep)
        for phrase in string_.split(sep):
            phrase = phrase.strip()
            
            while len(phrase) > SentenceTokenizer.MAXIMUM_SENTENCE_LENGTH:
                # clip too long sentences.
                sub_phrase = phrase[:SentenceTokenizer.MAXIMUM_SENTENCE_LENGTH].lstrip()
                splitted_string.append(sub_phrase)
                phrase = phrase[SentenceTokenizer.MAXIMUM_SENTENCE_LENGTH:].rstrip()
            
            if len(phrase) >= SentenceTokenizer.MINIMUM_SENTENCE_LENGTH:
                splitted_string.append(phrase)

        return splitted_string


with open("./data/How-to-Grow-Neat-Software-Architecture-out-of-Jupyter-Notebooks.md") as f:
    raw_data = f.read()

test_str_tokenized = SentenceTokenizer().fit_transform(raw_data)

# Print text example:
print(len(test_str_tokenized))
pprint(test_str_tokenized[3:9])
```

    168
    ["Have you ever been in the situation where you've got Jupyter notebooks "
     '(iPython notebooks) so huge that you were feeling stuck in your code?',
     'Or even worse: have you ever found yourself duplicating your notebook to do '
     'changes, and then ending up with lots of badly named notebooks?',
     "Well, we've all been here if using notebooks long enough.",
     'So how should we code with notebooks?',
     "First, let's see why we need to be careful with notebooks.",
     "Then, let's see how to do TDD inside notebook cells and how to grow a neat "
     'software architecture out of your notebooks.']


## Creating a SGNN preprocessing pipeline's classes


```python
class WordTokenizer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        begin_of_word = "<"
        end_of_word = ">"
        out = [
            [
                begin_of_word + word + end_of_word
                for word in sentence.replace("//", " /").replace("/", " /").replace("-", " -").replace("  ", " ").split(" ")
                if not len(word) == 0
            ]
            for sentence in X
        ]
        return out

```


```python
char_ngram_range = (1, 4)

char_term_frequency_params = {
    'char_term_frequency__analyzer': 'char',
    'char_term_frequency__lowercase': False,
    'char_term_frequency__ngram_range': char_ngram_range,
    'char_term_frequency__strip_accents': None,
    'char_term_frequency__min_df': 2,
    'char_term_frequency__max_df': 0.99,
    'char_term_frequency__max_features': int(1e7),
}

class CountVectorizer3D(CountVectorizer):

    def fit(self, X, y=None):
        X_flattened_2D = sum(X.copy(), [])
        super(CountVectorizer3D, self).fit_transform(X_flattened_2D, y)  # can't simply call "fit"
        return self

    def transform(self, X):
        return [
            super(CountVectorizer3D, self).transform(x_2D)
            for x_2D in X
        ]
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

```


```python
import scipy.sparse as sp

T = 80
d = 14

hashing_feature_union_params = {
    # T=80 projections for each of dimension d=14: 80 * 14 = 1120-dimensionnal word projections.
    **{'union__sparse_random_projection_hasher_{}__n_components'.format(t): d
       for t in range(T)
    },
    **{'union__sparse_random_projection_hasher_{}__dense_output'.format(t): False  # only AFTER hashing.
       for t in range(T)
    }
}

class FeatureUnion3D(FeatureUnion):
    
    def fit(self, X, y=None):
        X_flattened_2D = sp.vstack(X, format='csr')
        super(FeatureUnion3D, self).fit(X_flattened_2D, y)
        return self
    
    def transform(self, X): 
        return [
            super(FeatureUnion3D, self).transform(x_2D)
            for x_2D in X
        ]
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

```

## Fitting the pipeline 

Note: at fit time, the only thing done is to discard some unused char n-grams and to instanciate the random hash, the whole thing could be independent of the data, but here because of discarding the n-grams, we need to "fit" the data. Therefore, fitting could be avoided all along, but we fit here for simplicity of implementation using scikit-learn.


```python
params = dict()
params.update(char_term_frequency_params)
params.update(hashing_feature_union_params)

pipeline = Pipeline([
    ("word_tokenizer", WordTokenizer()),
    ("char_term_frequency", CountVectorizer3D()),
    ('union', FeatureUnion3D([
        ('sparse_random_projection_hasher_{}'.format(t), SparseRandomProjection())
        for t in range(T)
    ]))
])
pipeline.set_params(**params)

result = pipeline.fit_transform(test_str_tokenized)

print(len(result), len(test_str_tokenized))
print(result[0].shape)
```

    168 168
    (12, 1120)


## Let's see the output and its form. 


```python
print(result[0].toarray().shape)
print(result[0].toarray()[0].tolist())
print("")

# The whole thing is quite discrete:
print(set(result[0].toarray()[0].tolist()))

# We see that we could optimize by using integers here instead of floats by counting the occurence of every entry.
print(Counter(result[0].toarray()[0].tolist()))
```

    (12, 1120)
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.005715251142432, 0.0, 2.005715251142432, 0.0, 0.0, 2.005715251142432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    {0.0, 2.005715251142432, -2.005715251142432}
    Counter({0.0: 1069, -2.005715251142432: 27, 2.005715251142432: 24})


## Checking that the cosine similarity before and after word projection is kept

Note that this is a yet low-quality test, as the neural network layers above the projection are absent, so the similary is not yet semantic, it only looks at characters.


```python
word_pairs_to_check_against_each_other = [
    # Similar:
    ["start", "started"],
    ["prioritize", "priority"],
    ["twitter", "tweet"],
    ["Great", "great"],
    # Dissimilar:
    ["boat", "cow"],
    ["orange", "chewbacca"],
    ["twitter", "coffee"],
    ["ab", "ae"],
]

before = pipeline.named_steps["char_term_frequency"].transform(word_pairs_to_check_against_each_other)
after = pipeline.named_steps["union"].transform(before)

for i, word_pair in enumerate(word_pairs_to_check_against_each_other):
    cos_sim_before = cosine_similarity(before[i][0], before[i][1])[0,0]
    cos_sim_after  = cosine_similarity( after[i][0],  after[i][1])[0,0]
    print("Word pair tested:", word_pair)
    print("\t - similarity before:", cos_sim_before, 
          "\t Are words similar?", "yes" if cos_sim_before > 0.5 else "no")
    print("\t - similarity after :", cos_sim_after , 
          "\t Are words similar?", "yes" if cos_sim_after  > 0.5 else "no")
    print("")
```

    Word pair tested: ['start', 'started']
    	 - similarity before: 0.8728715609439697 	 Are words similar? yes
    	 - similarity after : 0.8542062410985866 	 Are words similar? yes
    
    Word pair tested: ['prioritize', 'priority']
    	 - similarity before: 0.8458888522202895 	 Are words similar? yes
    	 - similarity after : 0.8495862181305898 	 Are words similar? yes
    
    Word pair tested: ['twitter', 'tweet']
    	 - similarity before: 0.5439282932204212 	 Are words similar? yes
    	 - similarity after : 0.4826046482460216 	 Are words similar? no
    
    Word pair tested: ['Great', 'great']
    	 - similarity before: 0.8006407690254358 	 Are words similar? yes
    	 - similarity after : 0.8175049752615363 	 Are words similar? yes
    
    Word pair tested: ['boat', 'cow']
    	 - similarity before: 0.1690308509457033 	 Are words similar? no
    	 - similarity after : 0.10236537810666581 	 Are words similar? no
    
    Word pair tested: ['orange', 'chewbacca']
    	 - similarity before: 0.14907119849998599 	 Are words similar? no
    	 - similarity after : 0.2019908169580899 	 Are words similar? no
    
    Word pair tested: ['twitter', 'coffee']
    	 - similarity before: 0.09513029883089882 	 Are words similar? no
    	 - similarity after : 0.1016460166230715 	 Are words similar? no
    
    Word pair tested: ['ab', 'ae']
    	 - similarity before: 0.408248290463863 	 Are words similar? no
    	 - similarity after : 0.42850530886130067 	 Are words similar? no
    


## Next up

So we have created the sentence preprocessing pipeline and the sparse projection (random hashing) function. We now need a few feedforward layers on top of that. 

Also, a few things could be optimized, such as using the power set of the possible n-gram values with a predefined character set instead of fitting it, and the Hashing's fit function could be avoided as well by passing the random seed earlier, because the Hasher doesn't even look at the data and it only needs to be created at some point. This would yield a truly embedding-free approach. Free to you to implement this. I wanted to have something that worked first, leaving optimization for later.


## License


BSD 3-Clause License


Copyright (c) 2018, Guillaume Chevalier

All rights reserved.


## Extra links

### Connect with me

- [LinkedIn](https://ca.linkedin.com/in/chevalierg)
- [Twitter](https://twitter.com/guillaume_che)
- [GitHub](https://github.com/guillaume-chevalier/)
- [Quora](https://www.quora.com/profile/Guillaume-Chevalier-2)
- [YouTube](https://www.youtube.com/c/GuillaumeChevalier)
- [Dev/Consulting](http://www.neuraxio.com/en/)

### Liked this piece of code? Did it help you? Leave a [star](https://github.com/guillaume-chevalier/SGNN-Self-Governing-Neural-Networks-Projection-Layer/stargazers), [fork](https://github.com/guillaume-chevalier/SGNN-Self-Governing-Neural-Networks-Projection-Layer/network/members) and share the love!

