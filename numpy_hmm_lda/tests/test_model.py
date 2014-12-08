import logging
import unittest
import numpy as np

from numpy_hmm_lda.model import HiddenMarkovModelLatentDirichletAllocation, categorical

logging.basicConfig(level=logging.ERROR)

class TestModel(unittest.TestCase):

    def test_categorical(self):
        for i in range(16):
            counts = np.zeros(3, dtype=np.int64)
            for j in range(1000):
                counts[categorical(np.full(3, 2.0 ** i))] += 1
            print counts

    def test_generate(self):
        '''
        Start from hyperparameters and generate a document
        '''
        print
        num_topics = 4
        num_classes = 4
        num_documents = 16
        num_words = 64
        num_words_per_document = 64

        alpha_scalar = 0.25
        beta_scalar = 0.5
        gamma_scalar = 0.5
        delta_scalar = 0.5

        alpha = np.full(num_topics, alpha_scalar, dtype=np.float64)
        beta = np.full(num_words, beta_scalar, dtype=np.float64)
        gamma = np.full(num_classes, gamma_scalar, dtype=np.float64)
        delta = np.full(num_words, delta_scalar, dtype=np.float64)

        per_document_topic_distributions = np.random.dirichlet(alpha, num_documents)
        per_topic_word_distributions = np.random.dirichlet(beta, num_topics)
        transition_matrix = np.random.dirichlet(gamma, num_classes)
        per_class_word_distributions = np.random.dirichlet(delta, num_classes)

        sentence_start_class = 1

        def generate(document_idx):
            document = np.empty(num_words_per_document, dtype=np.int64)
            previous = None
            for word_idx in range(num_words_per_document):
                word_topic = np.random.multinomial(1, per_document_topic_distributions[document_idx])[0]
                if previous is not None:
                    word_class = np.random.multinomial(1, transition_matrix[previous])[0]
                else:
                    word_class = np.random.randint(num_classes)
                if word_class == 0:
                    word = np.random.multinomial(1, per_topic_word_distributions[word_topic])[0]
                else:
                    word = np.random.multinomial(1, per_class_word_distributions[word_class])[0]
                document[word_idx] = word
                previous = word
            return document

        model = HiddenMarkovModelLatentDirichletAllocation(vocab_size=num_words,
                                                           num_topics=num_topics,
                                                           num_classes=num_classes,
                                                           alpha=alpha_scalar,
                                                           beta=beta_scalar,
                                                           gamma=gamma_scalar,
                                                           delta=delta_scalar)

        for document_idx in range(num_documents):
            model.add_document(generate(document_idx))

        model.initialize()
        model.train(1)

        for idx, topic_distribution in enumerate(model.per_document_topic_distributions()):
            print '%d: %s %s' % (idx, np.round(topic_distribution, decimals=2), np.round(per_document_topic_distributions[idx], decimals=2))
