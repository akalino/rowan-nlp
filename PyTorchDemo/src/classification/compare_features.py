import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    @staticmethod
    def on_batch_end(self, model):
        print('batch end')

    def on_epoch_end(self, model):
        self.epoch += 1
        if self.epoch % 10 == 0:
            print('Completed {} epochs'.format(self.epoch))


def load_data():
    """
    Loads two classes of the Twenty Newsgroups dataset.
    :return: Sklearn dataset object.
    """
    categories = ['comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories,
                                      shuffle=True,
                                      random_state=42)
    print('Loaded {} documents'.format(len(twenty_train.data)))
    return twenty_train


def build_tfidf_features(_train):
    """
    Builds bag-of-words feature representations.
    :param _train: Training data.
    :return: TFxIDF features.
    """
    count_vecs = CountVectorizer()
    x_train_counts = count_vecs.fit_transform(_train.data)
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    return x_train_tfidf, count_vecs, tfidf_transformer


def build_w2v_features(_train, _in_dim):
    sentences = []
    features = []
    for doc in _train.data:
        sentences.append(doc.split(' ')) # for each document, split it and add to list
    # try different window sizes, documents are quite long
    model = Word2Vec(sentences=sentences,
                     vector_size=_in_dim,
                     window=20, min_count=5,
                     workers=1, seed=17)
    # try epochs 1, 10, 50, at 1000 accuracy matches TF x IDF
    epoch_log = EpochLogger()
    model.train(sentences, epochs=10,
                total_examples=model.corpus_count, callbacks=[epoch_log])
    for doc in tqdm(sentences):
        doc_vec = np.zeros(_in_dim)
        for tok in doc:
            try:
                doc_vec += model.wv[tok]
            except KeyError:
                pass
        features.append(doc_vec)
    features = np.asarray(features)
    return features, model


def build_simple_classifier(_features, _data):
    clf = LogisticRegression(max_iter=10000).fit(_features, _data.target)
    return clf


def evaluate_classifiers(_clf, _cv, _tft):
    categories = ['comp.graphics', 'sci.med']
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    test_cv = _cv.transform(twenty_test.data)
    test_tfidf = _tft.transform(test_cv)
    pred = _clf.predict(test_tfidf)
    acc = np.mean(pred == twenty_test.target)
    print('Logistic regression model with TF x IDF features has test accuracy {}'.format(acc))


def evaluate_wvs(_clf, _w2v, _in_dim):
    sentences = []
    features = []
    categories = ['comp.graphics', 'sci.med']
    _test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    for doc in _test.data:
        sentences.append(doc.split(' '))
    for doc in tqdm(sentences):
        doc_vec = np.zeros(_in_dim)
        for tok in doc:
            try:
                doc_vec += _w2v.wv[tok]
            except KeyError:
                pass
        features.append(doc_vec)
    features = np.asarray(features)
    pred = _clf.predict(features)
    acc = np.mean(pred == _test.target)
    print('Logistic regression model with Word2Vec features has test accuracy {}'.format(acc))


if __name__ == "__main__":
    FEATURE_DIM = 300
    train_ds = load_data()
    feats, cv, tft = build_tfidf_features(train_ds)
    print(feats.shape)

    clf = build_simple_classifier(feats, train_ds)
    evaluate_classifiers(clf, cv, tft)

    wv_feats, w2v = build_w2v_features(train_ds, FEATURE_DIM)
    print(wv_feats.shape)

    clf = build_simple_classifier(wv_feats, train_ds)
    evaluate_wvs(clf, w2v, FEATURE_DIM)

