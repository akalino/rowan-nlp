from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

ap = "C://Users//KALINOWA//PycharmProjects//rowan-nlp//pretrained//glove.6B.50d.txt"
glove_file = datapath(ap)
tmp_file = get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)