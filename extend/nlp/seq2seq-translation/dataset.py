
# coding: utf-8

# # 数据的读取和预处理
# 目标：产生源语言与目标语言的one-hot的tensor表示
# 
# tensor([[211, 212, 594,   5,   1]])
# tensor([[130, 125,  58,   4,   1]])

# In[22]:


import re
import os
import random
import torch
import unicodedata
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# In[23]:


SOS_token = 0                                   # 词汇表中的起止符
EOS_token = 1
MAX_LENGTH = 10


# In[24]:


# 定义词汇表类，保存词汇表
class Language(object):                         # 普通类继承于object
    def __init__(self, language):
        """
        language: the name of the language
        """
        self.language = language
        self.word2index = {'SOS':0,'EOS':1}  # the dict of the word with respect to index
        self.word2count = {}                 # the number of occurrences 
        self.index2word = {0:'SOS', 1:'EOS'} # index with respect to word,special token included
        self.n_words = 2                     # total number of the vocabulary
    def addSentence(self, sentence):
        for word in sentence.split(' '):     # split sentence by ' '
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:      # !!! not in dictionary's keys
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[20]:


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)       # add space between character and interpunction
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)   # only match the a-z/A-Z/.!?
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    file_path = os.path.join('../..', 'data', 'translation', '%s-%s.txt' % (lang1, lang2))
    lines = open(file_path, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Language(lang2)
        output_lang = Language(lang1)
    else:
        input_lang = Language(lang1)
        output_lang = Language(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = ("i am ", "i m ", "he is", "he s ", "she is", "she s",
                "you are", "you re ", "we are", "we re ", "they are",
                "they re ")


def filterPair(p):       # select the short sentence and which is started with the given prefixes
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):  # filter all pairs
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])               # add word to vocabulary
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.language, input_lang.n_words)
    print(output_lang.language, output_lang.n_words)
    print(random.choice(pairs))
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):              # sentence to indexs
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):               # sentence to tensor
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes)
    return result


def tensorFromPair(input_lang, output_lang, pair):    # pair to tensor
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


class TextDataset(Dataset):
    def __init__(self, dataload=prepareData, lang=['eng', 'fra']):
        """
        dataload: the dataset for dataloader
        lang: the name of the source and target language 
        """
        self.input_lang, self.output_lang, self.pairs = dataload(
            lang[0], lang[1], reverse=True)
        self.input_lang_words = self.input_lang.n_words
        self.output_lang_words = self.output_lang.n_words

    def __getitem__(self, index):
        return tensorFromPair(self.input_lang, self.output_lang,
                              self.pairs[index])

    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    da = TextDataset()
    lang_dataloader = DataLoader(da, shuffle=True)
    for data in lang_dataloader:
        print(data[0])
        print(data[1])
        break

