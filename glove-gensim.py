
"""
word2vec embeddings start with a line with the number of lines (tokens?) and
the number of dimensions of the file. This allows gensim to allocate memory
accordingly for querying the model. Larger dimensions mean larger memory is
held captive. Accordingly, this line has to be inserted into the GloVe
embeddings file.
"""

import os
import shutil
import hashlib
from sys import platform

import gensim


def prepend_line(infile, outfile, line):
    """
    Function use to prepend lines using bash utilities in Linux.
    (source: http://stackoverflow.com/a/10850588/610569)
    """
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    """
    Slower way to prepend the line by re-creating the inputfile.
    """
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def checksum(filename):
    """
    This is to verify the file checksum is the same as the glove files we use to
    pre-computed the no. of lines in the glove file(s).
    """
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()


def check_num_lines_in_glove(filename, check_checksum=False):
    if check_checksum:
        assert checksum(filename) == pretrain_checksum[filename]
    if filename.startswith('glove.6B.'):
        return 400000
    elif filename.startswith('glove.twitter.27B.'):
        return 1193514
    else:
        return pretrain_num_lines[filename]


if __name__ == "__main__":
    # Pre-computed glove files values.
    pretrain_num_lines = {"pretrained/glove.840B.300d.txt": 2196017,
                          "pretrained/glove.42B.300d.txt": 1917494,
                          "pretrained/glove.6B.300d.txt": 400000}

    pretrain_checksum = {
        "pretrained/glove.6B.300d.txt": "b78f53fb56ec1ce9edc367d2e6186ba4",
        "pretrained/glove.twitter.27B.50d.txt": "6e8369db39aa3ea5f7cf06c1f3745b06",
        "pretrained/glove.42B.300d.txt": "01fcdb413b93691a7a26180525a12d6e",
        "pretrained/glove.6B.50d.txt": "0fac3659c38a4c0e9432fe603de60b12",
        "pretrained/glove.6B.100d.txt": "dd7f3ad906768166883176d69cc028de",
        "pretrained/glove.twitter.27B.25d.txt": "f38598c6654cba5e6d0cef9bb833bdb1",
        "pretrained/glove.6B.200d.txt": "49fa83e4a287c42c6921f296a458eb80",
        "pretrained/glove.840B.300d.txt": "eec7d467bccfa914726b51aac484d43a",
        "pretrained/glove.twitter.27B.100d.txt": "ccbdddec6b9610196dd2e187635fee63",
        "pretrained/glove.twitter.27B.200d.txt": "e44cdc3e10806b5137055eeb08850569",
    }

    glove_file ="pretrained/glove.6B.300d.txt"
    _, tokens, dimensions, _ = glove_file.split('.')
    num_lines = check_num_lines_in_glove(glove_file)
    dims = int(dimensions[:-1])

    # Output: Gensim Model text format.
    gensim_file ='pretrained/gensim/glove_{t}_{d}_model.txt'.format(t=tokens, d=dimensions)
    gensim_first_line = "{} {}".format(num_lines, dims)

    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(glove_file, gensim_file, gensim_first_line)
    else:
        prepend_slow(glove_file, gensim_file, gensim_first_line)

    # Demo: Loads the newly created glove_model.txt into gensim API.
    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)

    print(model.most_similar(positive=['australia'], topn=10))
    print(model.similarity('woman', 'man'))
