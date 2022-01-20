#%%
import pandas as pd
import numpy as np
from nltk import wordpunct_tokenize
import regex
import re
import multiprocessing as mp
from nltk import wordpunct_tokenize, sent_tokenize

data_path = "../../"
output_path = "../output/"

def sent_tknzr(text):
    #text = re.sub(r'([a-z]{3})\.([A-Z])', r'\1 . \2', text) # why?
    sent_tokens = sent_tokenize(text)
    n_sentences = len(sent_tokens)
    return n_sentences, sent_tokens

#%%

# load articles from the HBR
data = pd.read_csv(data_path + "hbr_data.txt", sep='\t')
# create a new column to preserve original text
data["text"] = data["clean_text"].copy()

#%%

#=============================
# 1. Clean articles
#
# Before spliting the articles into sentences, we will use the formating information
# available to clean the article and extract some information (e.g. author, title)
#
# 1. Extract authors using: "\n By (or BY) THE NAME OF THE AUTHOR \n". Remove from text
# 2. Remove small (max 4 words) paragraphs with all caps
# 3. Remove small paragraphs that contain "harvard business review"
# 4. Remove small paragraphs with "... copyright ... "
# 5. Remove small paragraphs with only numbers
#=============================


# 1. Authors
data["authors"] = data["clean_text"].apply(lambda x: re.findall(r"\nBy [A-Z].{4,50}\n|\nBY [A-Z].{4,50}\n", x))
data["num_authors"] = data["authors"].apply(lambda x: len(x))
data["num_authors"].value_counts()
# remove
data["clean_text"] = data["clean_text"].apply(lambda x: re.sub(r"\nBy [A-Z].{4,50}\n|\nBY [A-Z].{4,50}\n", "\n", x))

# 2. All caps max 4 words
data["short_caps"] = data["clean_text"].apply(lambda x: re.findall(r"\n[A-Z]{1,20}\n|\n[A-Z]{1,20} [A-Z]{1,20}\n|\n[A-Z]{1,20} [A-Z]{1,20} [A-Z]{1,20}\n|\n[A-Z]{1,20} [A-Z]{1,20} [A-Z]{1,20} [A-Z]{1,20}\n", x))
data["clean_text"] = data["clean_text"].apply(lambda x: re.sub(r"\n[A-Z]{1,20}\n|\n[A-Z]{1,20} [A-Z]{1,20}\n|\n[A-Z]{1,20} [A-Z]{1,20} [A-Z]{1,20}\n|\n[A-Z]{1,20} [A-Z]{1,20} [A-Z]{1,20} [A-Z]{1,20}\n", "\n", x))

# 3. HBR
data["short_hbr"] = data["clean_text"].apply(lambda x: re.findall(r"\n.{0,10}harvard business review .{1,50}\n", x, flags=re.IGNORECASE))
data["num_hbr"] = data["short_hbr"].apply(lambda x: len(x))
data["num_hbr"].value_counts()
# remove
data["clean_text"] = data["clean_text"].apply(lambda x: re.sub(r"\n.{0,10}harvard business review .{1,50}\n", "\n", x, flags=re.IGNORECASE))

# 4. Copyright
data["copyright"] = data["clean_text"].apply(lambda x: re.findall(r"\n.{0,20}copyright.{0,80}\n", x, flags=re.IGNORECASE))
data["num_copy"] = data["copyright"].apply(lambda x: len(x))
data["num_copy"].value_counts()
# remove
data["clean_text"] = data["clean_text"].apply(lambda x: re.sub(r"\n.{0,20}copyright.{0,80}\n", "\n", x, flags=re.IGNORECASE))

# 5. Numbers
data["number"] = data["clean_text"].apply(lambda x: re.findall(r"\n.{0,5}[1-9].{0,5}\n", x, flags=re.IGNORECASE))
data["num_num"] = data["number"].apply(lambda x: len(x))
data["num_num"].value_counts()
# remove
data["clean_text"] = data["clean_text"].apply(lambda x: re.sub(r"\n.{0,5}[1-9].{0,5}\n", "\n", x, flags=re.IGNORECASE))

#%%

# take a look at some examples
i = np.random.randint(0, len(data)) # 11376
print(data.loc[i, "text"])
print("\n===================\n")
print(data.loc[i, "clean_text"])

#%%

#============================
# 2. Transform articles into sentences
#============================

files = []
dates = []
sentences = []
sent_index = []

for index, row in data.iterrows():

    temp_n, temp = sent_tknzr(row["clean_text"])

    sentences.extend(temp)
    sent_index.extend(list(range(temp_n)))
    files.extend(temp_n * [row['name']])
    dates.extend(temp_n * [row['date']])

    if(index % 250 == 0):
        print('completed iteration', index)

data_sent = pd.DataFrame({'name': files,
                          'sent_index': sent_index,
                          'date': dates,
                          'content': sentences})

# save final data
data_sent.to_csv(output_path + "data_sent_YM.txt", sep='\t', index=False)
#%%

# check random sentences
i = np.random.randint(0, len(data_sent))
print(data_sent.loc[i, "content"])

#%%

#========================
# 3. OLD: additional cleaning
#========================

# def hyphen_replace(text):
#     return(re.sub(r'\b-\b', ' ', text))

# data_sent['content'] = data_sent.content.apply(hyphen_replace)
# data_sent['content'] = data_sent.content.apply(lambda x: x.replace(u'“', '"'))
# data_sent['content'] = data_sent.content.apply(lambda x: x.replace(u'”', '"'))
# data_sent['content'] = data_sent.content.apply(lambda x: x.replace(u'’', '\''))
# data_sent['content'] = data_sent.content.apply(lambda x: x.replace(u'—', '-'))


# def token_clean(token):

#     '''
#     determine whether a token forms part of a "standard" sentence
#     '''

#     if token[0].isalpha() and token[1:].islower():
#         return(1)
#     elif len(token) == 1 and token.isalpha():
#         return(1)
#     elif token.isnumeric():
#         return(1)
#     elif regex.search(r'\p{Sc}', token):
#         return(1)
#     elif regex.search(r'[?!\.:;\',"()%-]', token):
#         # dash should be last (or 1st)
#         return(1)
#     else:
#         return(0)


# def clean_sentence_flag(text):

#     tokens = wordpunct_tokenize(text)
#     flags = [token_clean(t) for t in tokens]
#     if min(flags) == 0:
#         return(True)
#     else:
#         return(False)


# with mp.Pool() as pool:
#     flags = pool.map(clean_sentence_flag, data_sent.content)

# data_sent_clean = data_sent[~pd.Series(flags)]
# data_sent_clean.reset_index(drop=True, inplace=True)
# data_sent_dropped = data_sent[pd.Series(flags)]
# data_sent_dropped.reset_index(drop=True, inplace=True)

# save data
#data_sent_clean.to_csv("data_sent.txt", sep='\t', index=False)
#data_sent_dropped.to_csv("data_dropped.txt", sep='\t', index=False)

# %%

