import numpy as np
import pandas as pd
from .TermWeighting import GetTfIdf, Tokenizer, Term, Frequency, TextPreprocessing
import os
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
stemming_dir = os.path.abspath(os.path.join(base_dir, 'datasets/hasil_stemming.xlsx'))
best_train = os.path.abspath(os.path.join(base_dir, 'datasets/best_train.xlsx'))

stemming_alquran = np.array(pd.read_excel(stemming_dir)[0])
centroid = pd.read_excel(best_train, sheet_name='centroid').drop('cluster', axis='columns')
alquran = pd.read_excel(best_train, sheet_name='quran')
print("Data terload")
tf_idf_all_surah = GetTfIdf(stemming_alquran)
print("Sukses menghitung TF-IDF")



def Execute(query):
    df_tf_idf = pd.DataFrame(tf_idf_all_surah)
    concat_tf_idf_alquran = pd.concat([alquran, df_tf_idf], axis = 1)

    stemming_quran = stemming_alquran


    # <Doc Freq>
    data_tokenizing = Tokenizer(stemming_quran)
    data_term_unique = Term(data_tokenizing)
    doc_freq = np.sum(np.array(Frequency(data_tokenizing, data_term_unique)).T, axis=1)

    query_term, query_freq = TextPreprocessing(query)

    freq = []
    index = []
    q_freq = []
    for i in range(len(query_term)):
        cari_index = np.where(data_term_unique == query_term[i])
        if cari_index[0]: 
            cari_index = cari_index[0][0]
            index.append(cari_index)
            freq.append(doc_freq[cari_index])
            q_freq.append(query_freq[i])
    freq = np.array(freq)
    index = np.array(index)
    query_freq = np.array(q_freq)
    # </Doc Freq>

    tf_idf_square = (1 + np.log10(query_freq)) * np.log10(len(tf_idf_all_surah)/freq)**2
    norm = tf_idf_square / np.sqrt(np.sum(tf_idf_square))

    array_kosong = np.zeros(len(data_term_unique))
    for i in range(len(index)):
        array_kosong[index[i]] = norm[i]
    norm_query = array_kosong

    euclid = np.sqrt(np.sum((np.array(centroid) - norm_query)**2, axis=1))
    kelas = np.argmax(euclid)

    data_classnya = concat_tf_idf_alquran.loc[concat_tf_idf_alquran['label'] == kelas]
    meaning = np.array(data_classnya[['type', 'surat', 'ayat', 'arti', 'arab']])
    tf_idf_data = np.array((data_classnya))[:,6:]


    ir = np.sum(norm_query * tf_idf_data, axis=1)
    sort_ir = np.sort(np.unique(ir))[::-1]

    hasil_ir = []
    for i in sort_ir:
        if i <= 0:
            break
        index_ke = np.where(ir == i)
        hasil_ir.append(meaning[index_ke])

    pop = []
    hasil_ir = np.array(hasil_ir)
    for i in hasil_ir:
        for j in i:
            pop.append(j)
    pop = np.array(pop[:25])

    return pop
