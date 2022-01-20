#%%
import pandas as pd
import numpy as np
import sys
import string
import json
import re
from nltk import wordpunct_tokenize, sent_tokenize

sys.path.append('./pymodules')
import preprocessing_class as pc

data_path = "./"
output_path = "./"
dicts_path = "./original_code/phrases_hbr/"

def phrase_make(text, suffix):
    temp = wordpunct_tokenize(text)
    temp = [t for t in temp if t.isalpha()]
    temp = '-'.join(temp)
    temp = temp + suffix
    return(temp)

def sent_clean(text):
    text = text.lower()
    text = re.sub(r"[' ']*([&.,;:?!\(\)\[\]])[' ']*", r' \1 ', text)
    text = re.sub(r"[' ']*[-—'’\"/\\+*=^%#@~]+[' ']*", ' ', text)
    text = re.sub(r"[' ']*u \. s \.[' ']*", r' us ', text)
    text = re.sub(r"[' ']+gases[' ']+", r' gas ', text)
    tokens = wordpunct_tokenize(text)
    return ' '.join(tokens)

# %%

#==========================
# 1. Build replacement dictionaries
#==========================

# BUILD DICTIONARY FOR MANAGEMENT CONCEPTS
with open(dicts_path + "mng_dicts.txt") as f:
    mng_dicts = json.loads(f.read())

replace_dict_mng = {}

for k in mng_dicts.keys():
    replace_dict_mng[k] = phrase_make(k, '-mng')
    for t in mng_dicts[k]:
        replace_dict_mng[t] = phrase_make(t, '-mng')

replace_dict_mng = {k: phrase_make(k, '-mng') for k in mng_dicts.keys()}


# BUILD DICTIONARY AND FUNCTION FOR NAMED ENTITIES
entities = pd.read_csv(dicts_path + "named_entities.txt",
                       sep='\t', header=None)

entities['N'] = entities[0].apply(lambda x: len(x.split()))
entities = entities[(entities['N'] > 1) & (entities[1] > 10)]

replace = list(entities[0])
replace = list(set(replace))
replace = [x.strip() for x in replace]
replace = [sent_clean(x) for x in replace]

replace_dict_ent = {k: phrase_make(k, '-ent') for k in replace}

# BUILD DICTIONARY AND FUNCTION FOR NGRAMS
ngrams = pd.read_csv(dicts_path + 'ngrams_pre_select.csv')
ngrams['N'] = ngrams['ngram'].apply(lambda x: len(x.split()))

four_grams = list(ngrams[ngrams.N == 4]['ngram'])
replace = [sent_clean(x) for x in four_grams]
replace_dict_4 = {k: phrase_make(k, '') for k in replace}

tri_grams = list(ngrams[ngrams.N == 3]['ngram'])
replace = [sent_clean(x) for x in tri_grams]
replace_dict_3 = {k: phrase_make(k, '') for k in replace}

bi_grams = list(ngrams[ngrams.N == 2]['ngram'])
replace = [sent_clean(x) for x in bi_grams]
replace_dict_2 = {k: phrase_make(k, '') for k in replace}

# combine all dictionaries
replacing_dict = {**replace_dict_mng, 
                  **replace_dict_ent,
                  **replace_dict_2,
                  **replace_dict_3,
                  **replace_dict_4}

#%%

#================================
# 2. Apply Preprocessing
#================================

# define tokenization pattern and punctuation symbols
pattern = r'''
          (?x)                # set flag to allow verbose regexps (to separate logical sections of pattern and add comments)
          \w+(?:-\w+)*        # word characters with internal hyphens
          | [][.,;"'?():-_`]  # preserve punctuation as separate tokens
          '''

# remove hyphen from list of punctuation (we want to preserve it)
punctuation = string.punctuation.replace("-", "")

# iterate over all sentences from the HBR
df_complete = pd.DataFrame()
for i, chunk in enumerate(pd.read_csv(data_path + "data_sent_YM.txt",
                                      sep='\t',
                                      chunksize=100000)):
    
    print(f"Analyzing chunk: {i}")
    df = chunk

    # correct the words that were separated by line breaks
    df["content"] = df["content"].apply(lambda x: re.sub(r"-\n", "", x))
    
    # preprocess using the class
    prep = pc.RawDocs(df["content"], stopwords="short", lower_case=True, contraction_split=True, tokenization_pattern=pattern)
    prep.phrase_replace(replace_dict=replacing_dict, items='tokens', case_sensitive_replacing=False)
    # "length" determines the minimum size of a token
    prep.token_clean(length=2, punctuation=punctuation, numbers=True)
    prep.stopword_remove(items="tokens")

    # save result
    df["tokens"] = prep.tokens
    df_complete = pd.concat([df_complete, df[["name", "sent_index", "date", "content" ,"tokens"]]])
    
# calculate the number of tokens per sentence
df_complete["num_tokens"] = df_complete["tokens"].apply(lambda x: len(x))
#%%

# remove sentences with less than N tokens
N = 3
df_complete = df_complete.loc[df_complete["num_tokens"] > N]
df_complete.reset_index(drop=True, inplace=True)

# save results as pickle
df_complete = df_complete[["name", "sent_index", "date", "content", "tokens"]]
df_complete.to_pickle(output_path + "data_sent_clean_YM.pkl")

#%%

# check random sentences
i = np.random.randint(0, len(df_complete))
print(df_complete.loc[i, "content"])
print("\n------------------------\n")
print(df_complete.loc[i, "tokens"])

# %%

df_complete = pd.read_pickle("data_sent_clean_YM.pkl")
# %%
