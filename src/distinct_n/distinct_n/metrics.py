from distinct_n.distinct_n.utils import ngrams

__all__ = ["distinct_n_sentence_level", "distinct_n_corpus_level"]


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    # distinct_ngrams = set(ngrams(sentence, n))
    # print(ngrams(sentence, n))
    return list(set(ngrams(sentence, n)))
    # return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    temp = []
    length = 0
    for sentence in sentences:
        length += len(sentence)
        temp.extend(distinct_n_sentence_level(sentence, n))
    return len(set(temp)) / length
