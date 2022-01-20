#%%
import pandas as pd
import numpy as np
from nltk import sent_tokenize
import sys
import codecs
import string

# add path to source code for preprocessing
sys.path.append('./src')
import preprocessing_class as pc

data_path = "./data/"

# %%

# load sample of IMDB movie reviews
df = pd.read_csv(data_path + "imdb_sample.csv")
# create a new column to identify the review (an ID)
df.reset_index(inplace=True)
# %%

#============================
# 1. Sentence tokenization
#============================

review_index = []
sentences = []
sent_index = []

for index, row in df.iterrows():

    # apply sentence tokenization
    temp_sents = sent_tokenize(row["review"])
    temp_n = len(temp_sents)

    # save results
    sentences.extend(temp_sents)
    sent_index.extend(list(range(temp_n)))
    review_index.extend(temp_n * [row['index']])

    if(index % 250 == 0):
        print('completed iteration', index)

# consolidate all the sentences generated
df_sents = pd.DataFrame({'review_index': review_index,
                         'sent_index': sent_index,
                         'sentence': sentences})
# %%

#============================
# 2. Sentence preprocessing
#============================

# load list of stopwords
with codecs.open('./auxiliary/stopwords_long.txt', 'r', 'utf-8') as f:
    stp_long = list(f.read().splitlines())

# build replacement dictionary
replacing_dict = {'monetary policy':'monetary-policy',
                  'interest rate':'interest-rate',
                  'yield curve':'yield-curve',
                  'repo rate':'repo-rate',
                  'bond yields':'bond-yields',
                  'real estate':'real-estate',
                  'economic growth':'economic-growth'}

# define tokenization pattern and punctuation symbols
pattern = r'''
          (?x)                # set flag to allow verbose regexps (to separate logical sections of pattern and add comments)
          \w+(?:-\w+)*        # word characters with internal hyphens
          | [][.,;"'?():-_`]  # preserve punctuation as separate tokens
          '''

# define punctuation symbols to be removed
# remove hyphen from list of punctuation (we want to preserve it!)
punctuation = string.punctuation.replace("-", "")

#### 1. Create an instance of the preprocessing class
prep = pc.RawDocs(df_sents["sentence"],    # define corpus
                  stopwords=stp_long,      # define list with stopwords
                  lower_case=True,         # apply lowercasing
                  contraction_split=True,  # split English contractions
                  tokenization_pattern=pattern  # use pre-defined pattern
                  )

# initializing the class will automatically tokenize the corpus
print(prep.tokens[0])

#%%

#### 2. Replace phrases using the replacement dictionary
prep.phrase_replace(replace_dict=replacing_dict, 
                    items='tokens', 
                    case_sensitive_replacing=False)

#### 3. Apply a standard set of cleaning steps to the tokens
prep.token_clean(length=2,    # determines the minimum size of a token
                 punctuation=punctuation, 
                 numbers=True)

#### 3. Remove stopwords from tokens
prep.stopword_remove(items="tokens")

#### 4. Save resulting tokens as a column
df_sents["tokens"] = prep.tokens

#### 5. Minimum length per sentence
N = 3
# calculate the number of tokens per sentence
df_sents["num_tokens"] = df_sents["tokens"].apply(lambda x: len(x))
# remove sentences with less than N tokens
df_sents = df_sents.loc[df_sents["num_tokens"] > N]
df_sents.reset_index(drop=True, inplace=True)


# %%

#============================
# 3. Exploration
#============================

# check random sentences
i = np.random.randint(0, len(df_sents))
print(df_sents.loc[i, "sentence"])
print("\n------------------------\n")
print(df_sents.loc[i, "tokens"])
# %%
