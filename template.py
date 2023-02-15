import random

from collections import defaultdict

from process_corpus import preprocessing


def unigrams(_sents):
    """
    Computes unsmoothed unigram estimates.
    :param _sents: List of lists of tokens.
    :return: Dict, with unsmoothed unigram probabilities.
    """
    counts = defaultdict(int)
    probs = defaultdict(float)

    return None



def bigrams(_sents, _uni_counts):
    """
    Computes unsmoothed bigram probabilities.
    :param _sents: _sents: List of lists of tokens.
    :param _uni_counts: Unigram probability counts.
    :return: Dicts, counts and probabilities.
    """
    bigram_prob = defaultdict(float)
    return None


def laplace_unigrams(_sents, _smooth):
    """
    Computes unsmoothed unigram estimates.
    :param _sents: List of lists of tokens.
    :param _smooth: If True, run Laplace smoothing, otherwise unsmoothed.
    :return: Dict, with unsmoothed unigram probabilities.
    """
    counts = defaultdict(int)
    probs = defaultdict(float)
    return None


def laplace_bigrams(_sents, _uni_counts, _smooth):
    """
    Computes unsmoothed bigram probabilities.
    :param _sents: _sents: List of lists of tokens.
    :param _uni_counts: Unigram probability counts.
    :param _smooth: If True, run Laplace smoothing, otherwise unsmoothed.
    :return: Dicts, counts and probabilities.
    """
    bigram_prob = defaultdict(float)
    vocab_size = len(list(set(list(_uni_counts.keys()))))
    return None


def random_sentence(_bigrams, _max_length):
    random.seed(12)
    sent = ['<s>']
    eos = '<\\s>'
    return None


def compute_perplexity(_sent, _bigrams):
    """
    Computes the sample perplexity of a bigram model.

    :param _sent: an input sentence as a list of tokens.
    :param _bigrams: The bigram language model.
    :return: perplexity score.
    """
    return None


if __name__ == "__main__":
    sents = preprocessing('hamlet.txt')
    uni_count, uni_prob = unigrams(sents)
    uni_counts, uni_probs = laplace_unigrams(sents, True)
    bigram_probs = bigrams(sents, uni_count)
    bigram_probss = laplace_bigrams(sents, uni_counts, True)
    rs = random_sentence(bigram_probs, 12)
    perp = compute_perplexity(rs, bigram_probs)
    print(rs, perp)
    rs = random_sentence(bigram_probss, 12)
    perp = compute_perplexity(rs, bigram_probss)
    print(rs, perp)
