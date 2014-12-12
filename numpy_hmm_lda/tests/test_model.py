import logging
import unittest
import numpy as np

from numpy_hmm_lda.model import HiddenMarkovModelLatentDirichletAllocation, categorical

class TestModel(unittest.TestCase):

    def setUp(self):
        print

    def test_categorical(self):
        counts = np.zeros(2, dtype=np.int64)
        for i in range(16):
            for j in range(1000):
                counts[categorical(np.array([2.0 ** float(i), 2.0 ** float(i - 1)]))] += 1
        ratio = float(counts[0]) / float(counts[1])
        assert ratio > 1.75 and ratio < 2.25

    def test_initialize_progressive(self):
        num_topics = 4
        num_classes = 4
        num_documents = 16
        num_words = 64
        num_words_per_document = 256

        model = HiddenMarkovModelLatentDirichletAllocation(vocab_size=num_words,
                                                           num_topics=num_topics,
                                                           num_classes=num_classes,
                                                           alpha=0.0,
                                                           beta=0.0,
                                                           gamma=0.0,
                                                           delta=0.0)

        for document_idx in range(num_documents):
            model.add_document(np.random.randint(num_words, size=num_words_per_document))

        model.initialize_progressive()
        model.check()
