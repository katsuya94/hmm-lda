import logging
import unittest
import numpy as np

from numpy_hmm_lda.model import HiddenMarkovModelLatentDirichletAllocation

class TestModel(unittest.TestCase):

	def test_run(self):
		'''
		0 - syntactic word 1
		1 - syntactic word 2
		2 - semantic word A
		3 - semantic word B
		'''
		logging.basicConfig(level=logging.INFO)
		document_a = [np.array([0, 2]), np.array([1, 2])]
		document_b = [np.array([1, 3]), np.array([0, 3])]
		model = HiddenMarkovModelLatentDirichletAllocation(vocab_size=4,
														   num_topics=2,
														   num_classes=2,
														   alpha=1.0,
														   beta=1.0,
														   gamma=1.0,
														   delta=1.0)
		model.add_document(document_a)
		model.add_document(document_b)
		model.initialize()
		for idx, topic_distribution in enumerate(model.per_document_topic_distributions()):
			logging.info('%d: %s', idx, topic_distribution)
		model.train(100)
		for idx, topic_distribution in enumerate(model.per_document_topic_distributions()):
			logging.info('%d: %s', idx, topic_distribution)

