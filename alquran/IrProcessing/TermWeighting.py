import numpy as np
import pandas as pd
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import normalize
import re

import warnings
warnings.filterwarnings("ignore")

def TextPreprocessing(data):
    remove_char = re.sub(r'[^\w]', ' ', data)
    remove_number = "".join(filter(lambda x: not x.isdigit(), remove_char))
    to_lower = remove_number.lower()

    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    remove_stopword = stopword.remove(to_lower)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemming = stem = stemmer.stem(remove_stopword)
    tokening = stemming.split()
    term = np.unique(tokening)

    counting_freq = Counter(term)
    freq = []
    for i in term:
        freq.append(counting_freq[i])
    freq = np.array(freq)

    return term, freq

def Tokenizer(data):
    token = []
    for i in range(len(data)):
        tokening = data[i].split()
        token.append(tokening)
    # print(token)
    return token

def Term(data):
    data_to_one_d = [_ for i in range(len(data)) for _ in data[i]]
    term = np.unique(data_to_one_d)
    return term

def Frequency(data, term_unique):
    counter_freq = []
    for i in range(len(data)):
        counting_freq = Counter(data[i])
        # print(counting_freq)
        counter_freq.append(counting_freq)
    counter_freq = np.array(counter_freq)

    raw_term_freq = []
    for i in range(len(counter_freq)):
        doc = []
        for j in term_unique:
            if j in counter_freq[i]:
                value = counter_freq[i][j]
            else:
                value = 0
            doc.append(value)
        raw_term_freq.append(doc)
    return raw_term_freq

def LogFrequency(data):
    log_freq = np.where((data>0), (np.log10(data)+1),0)
    return log_freq

def DocumentFrequencyOfTerm(data):
    doc_freq = (data != 0).sum(1)
    return doc_freq

def InverseDocumentFrequencyOfTerm(data, log_frequency):
    inverse_doc_frec = np.log10(len(log_frequency[0])/data)
    return inverse_doc_frec

def TfIdfWeighting(log_frequency, inverse_document_frequency_of_term):
    count = log_frequency.T*inverse_document_frequency_of_term
    return count

def Normalize(data):
    a = np.sqrt(np.sum(data))
    norm = data / a
    return norm

def GetTfIdf(data):
  tokenizing = Tokenizer(data)
  term_unique = Term(tokenizing)
  trem_frequency = np.array(Frequency(tokenizing, term_unique)).T
  log_frequency = LogFrequency(trem_frequency)
  document_frequency_of_term = DocumentFrequencyOfTerm(log_frequency)
  inverse_document_frequency_of_term = InverseDocumentFrequencyOfTerm(document_frequency_of_term, log_frequency)
  tf_idf_wighting = TfIdfWeighting(log_frequency, inverse_document_frequency_of_term)
  normalisasi_tfidf = Normalize(tf_idf_wighting)
  return normalisasi_tfidf