import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import MeCab
import sys
import json
import collections	
from gensim import  models, corpora
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from tqdm import tqdm
from gensim.models import KeyedVectors



# 名詞に絞った分かち書きをする
def noun_extraction(sentence, mecab):
    mecab.parse("")
    #分割して、名詞をnounにそれぞれ格納
    noun_list = []
    node = mecab.parseToNode(sentence)
    while node:
        if re.search("^(名詞)", node.feature):
            noun_list.append(node.surface)
        node = node.next
    # 数詞表現の削除
    valid = remove_numerals(noun_list)
    # 辞書に含まれる語のみ許す
    return valid

def remove_numerals(word_list):
    valid = []
    for word in word_list:
        if isNumeral(word):
            pass
        else:
            valid.append(word)
    return valid

def isNumeral(value):
    # 数字から始まる単語を判定する
    return re.match(r"\d", value) is not None


def decompose_2d(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    X2d = pca.transform(X)
    return X2d

def create_labels(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters).fit(X)
    labels = kmeans.labels_
    return labels

def convert_corpus(corpus_tfidf, dictionary, num_topics ,method="LSI"):
    if method == "LDA":
        model = models.ldamodel.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    elif method == "LSI":
        model = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics = num_topics)
    tmp_corpus = model[corpus_tfidf]
    n = len(tmp_corpus)
    m = num_topics
    X = np.zeros((n,m))
    for i, lis in enumerate(tmp_corpus):
        for j,val in lis:
            X[i,j] = val
    return X

def plot_dendrogram(X):
    Z = hierarchy.linkage(X, 'single')
    plt.figure(figsize=(10,10))
    dn = hierarchy.dendrogram(Z)
    plt.savefig("./dendrogram.png")
    # plt.show()
    return Z

def search_num_clusters(Z):
    pre=0
    diffs = np.array([])
    for v in Z[:, 2]:
        diff = v-pre
        diffs = np.append(diffs, diff)
        pre = v
    n_cluster = len(diffs)-diffs.argmax()+1
    return n_cluster

def plot_scatter_clusters(X2d, labels, id_list):
    plt.figure(figsize = (20,20))
    plt.scatter(X2d[:,0], X2d[:,1], c = labels, s = 50, alpha=0.7)
    for i,s in enumerate(id_list):
        plt.text(X2d[:,0][i], X2d[:,1][i], s, fontsize=10)
    plt.savefig("./clusters.png")
    # plt.show()

def collect_keywords(texts, labels):
    allocated_texts = {label: np.array(texts)[np.where(labels==label)[0]] for label in list(set(labels))}

    allocated_words = {label: [word for text in mini_texts for word in text] for label, mini_texts in allocated_texts.items()}

    allocated_vocab = {label: set(words) for label, words in allocated_words.items()}

    allocated_unique_vocab = create_unique(allocated_vocab)

    freq_words = []
    for label in allocated_words.keys():
        words = allocated_words[label]
        vocab = allocated_unique_vocab[label]
        c = collections.Counter(words)
        dic = {}
        for word in vocab:
            dic[word] = c[word]
        freq100 = sorted(dic.items(), key=lambda x:x[1], reverse=True)[:100]
        freq_words.append(freq100)
    return freq_words

# 　他のlabelのテキスト群に出現したか否かを判定
def create_unique(allocated_vocab):
    allocated_unique_vocab = {}
    label_list = list(allocated_vocab.keys())
    
    for label, vocab in allocated_vocab.items():
        k = label_list.index(label)
        other_labels = label_list[:k] + label_list[k+1:]
        other_vocab = set()
        for ol in other_labels:
            other_vocab = other_vocab.union(allocated_vocab[ol])

        unique_vocab = set()
        for word in vocab:
            if word in other_vocab:
                pass
            else:
                unique_vocab.add(word)
        allocated_unique_vocab[label] = unique_vocab
    return allocated_unique_vocab
   
def listup_keywords(keywords):
    keywords_dict = {
    i+1 : keywords[i] for i in range(len(keywords))
    }
    with open("./keywords.json", 'w') as f_out:
        json.dump(keywords_dict,f_out,indent=4, ensure_ascii=False)



if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    csv_path =  sys.argv[1]
    num_topics = int(sys.argv[2])

    df = pd.read_csv(csv_path)

    # 辞書オブジェクトの読み込み
    mecab = MeCab.Tagger()

    print("Wakati Process....")
    # 分かち書きして，空白区切のstringで文字列を再現
    texts = []
    for rawtext in tqdm(df["doc_text"]):
        wakati = noun_extraction(rawtext, mecab)
        texts.append(wakati)
    id_list =  list(df["doc_id"])

    del df
    # 単語->id変換の辞書作成
    print("Create corpus......")
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.3)
    # textsをcorpus化
    corpus = list(map(dictionary.doc2bow,texts))
    # tfidf modelの生成
    test_model = models.TfidfModel(corpus)
    # corpusへのモデル適用
    corpus_tfidf = test_model[corpus]

    print("Decompose By Topic Model......")
    print("\tdimention: {} --> {}".format(len(dictionary), num_topics))
    X = convert_corpus(corpus_tfidf, dictionary, num_topics, "LSI")

    del corpus, dictionary, test_model, corpus_tfidf

    print("Create Dendrogram......")
    Z = plot_dendrogram(X)

    # クラスター数の最適化
    num_clusters = search_num_clusters(Z)
    print("Best num_clusters: ", num_clusters)
    labels = create_labels(X, num_clusters)


    # 主成分分析し，2次元でプロット
    X2d = decompose_2d(X)
    plot_scatter_clusters(X2d, labels, id_list)

    # キーワードを選出
    keywords = collect_keywords(texts, labels)
    listup_keywords(keywords)
