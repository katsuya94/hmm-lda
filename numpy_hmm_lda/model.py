import logging
import numpy as np

class HiddenMarkovModelLatentDirichletAllocation(object):

    def __init__(self, vocab_size=None, num_topics=None, num_classes=None, alpha=None, beta=None, gamma=None, delta=None):
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.num_classes = num_classes

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.documents = []

    def set_vocab_size(self, size):
        '''
        Sets the size of the vocabulary.
        size = integer vocabulary size
        '''
        self.vocab_size = size

    def add_document(self, document):
        '''
        Adds a document to the model.
        document = list of numpy arrays (sentences) of integers representing words
        '''
        logging.info('Adding document of %d sentences', len(document))
        self.documents.append(document)

    def initialize(self):
        '''
        Set random assignments of topics and classes initially.
        '''
        self.topic_assignments = [[np.random.random_integers(0, self.num_topics - 1, size=sentence.size) for sentence in document] for document in self.documents]
        self.class_assignments = [[np.random.random_integers(0, self.num_classes - 1, size=sentence.size) for sentence in document] for document in self.documents]
        self.num_words_in_doc_assigned_to_topic = [self.count_num_words_in_doc_assigned_to_topic(document_idx) for document_idx in range(len(self.documents))]
        self.num_same_words_assigned_to_topic = [self.count_num_same_words_assigned_to_topic(word) for word in range(self.vocab_size)]
        self.num_words_assigned_to_topic = self.count_num_words_assigned_to_topic()
        self.num_same_words_assigned_to_class = [self.count_num_same_words_assigned_to_class(word) for word in range(self.vocab_size)]
        self.num_words_assigned_to_class = self.count_num_words_assigned_to_class()
        self.num_transitions = self.count_num_transitions()

    def count_num_words_in_doc_assigned_to_topic(self, document_idx):
        topic_counts = np.zeros(self.num_topics, dtype=np.int32)
        for sentence_idx in range(len(self.documents[document_idx])):
            for word_idx in range(self.documents[document_idx][sentence_idx].size):
                if self.class_assignments[document_idx][sentence_idx][word_idx] == 0:
                    topic_counts[self.topic_assignments[document_idx][sentence_idx][word_idx]] += 1
        return topic_counts

    def count_num_same_words_assigned_to_topic(self, word):
        topic_counts = np.zeros(self.num_topics, dtype=np.int32)
        for document_idx in range(len(self.documents)):
            for sentence_idx in range(len(self.documents[document_idx])):
                for word_idx in range(self.documents[document_idx][sentence_idx].size):
                    if self.documents[document_idx][sentence_idx][word_idx] == word and self.class_assignments[document_idx][sentence_idx][word_idx] == 0:
                        topic_counts[self.topic_assignments[document_idx][sentence_idx][word_idx]] += 1
        return topic_counts

    def count_num_words_assigned_to_topic(self):
        topic_counts = np.zeros(self.num_topics, dtype=np.int32)
        for document_idx in range(len(self.documents)):
            for sentence_idx in range(len(self.documents[document_idx])):
                for word_idx in range(self.documents[document_idx][sentence_idx].size):
                    if self.class_assignments[document_idx][sentence_idx][word_idx] == 0:
                        topic_counts[self.topic_assignments[document_idx][sentence_idx][word_idx]] += 1
        return topic_counts

    def count_num_same_words_assigned_to_class(self, word):
        class_counts = np.zeros(self.num_classes, dtype=np.int32)
        for document_idx in range(len(self.documents)):
            for sentence_idx in range(len(self.documents[document_idx])):
                for word_idx in range(self.documents[document_idx][sentence_idx].size):
                    if self.documents[document_idx][sentence_idx][word_idx] == word:
                        class_counts[self.class_assignments[document_idx][sentence_idx][word_idx]] += 1
        return class_counts

    def count_num_words_assigned_to_class(self):
        class_counts = np.zeros(self.num_classes, dtype=np.int32)
        for document_idx in range(len(self.documents)):
            for sentence_idx in range(len(self.documents[document_idx])):
                for word_idx in range(self.documents[document_idx][sentence_idx].size):
                    class_counts[self.class_assignments[document_idx][sentence_idx][word_idx]] += 1
        return class_counts

    def count_num_transitions(self):
        transition_counts = np.zeros((self.num_classes, self.num_classes,), dtype=np.int32)
        for document_idx in range(len(self.documents)):
            for sentence_idx in range(len(self.documents[document_idx])):
                previous = None
                for word_idx in range(self.documents[document_idx][sentence_idx].size):
                    current = self.class_assignments[document_idx][sentence_idx][word_idx]
                    if previous is not None:
                        transition_counts[previous, current] += 1
                    previous = current

                    
        return transition_counts

    def train(self, iterations):
        '''
        Run Gibbs sampling
        iterations = number of full samplings of the corpus
        '''
        for i in range(iterations):
            for document_idx in range(len(self.documents)):
                for sentence_idx in range(len(self.documents[document_idx])):
                    previous = None
                    for word_idx in range(self.documents[document_idx][sentence_idx].size):
                        logging.info('[%d][%d][%d] = %d', document_idx, sentence_idx, word_idx, self.documents[document_idx][sentence_idx][word_idx])
                        logging.info('topic = %d', self.topic_assignments[document_idx][sentence_idx][word_idx])
                        logging.info('About to draw topic')
                        self.draw_topic(document_idx, sentence_idx, word_idx)
                        logging.info('class = %d', self.class_assignments[document_idx][sentence_idx][word_idx])
                        logging.info('About to draw class')
                        self.draw_class(document_idx, sentence_idx, word_idx)

    def draw_topic(self, document_idx, sentence_idx, word_idx):
        old_topic = self.topic_assignments[document_idx][sentence_idx][word_idx]
        word = self.documents[document_idx][sentence_idx][word_idx]

        # Initialize probability proportions with document topic counts

        proportions = self.num_words_in_doc_assigned_to_topic[document_idx].astype(np.float32)

        # Exclude current word

        proportions[old_topic] -= 1.0

        # Smoothing

        proportions += self.alpha

        # If the current word is assigned to the semantic class

        if self.class_assignments[document_idx][sentence_idx][word_idx] == 0:

            # Initialize numerator with same word topic counts

            numerator = self.num_same_words_assigned_to_topic[self.documents[document_idx][sentence_idx][word_idx]].astype(np.float32)

            # Initialize denominator with global topic counts

            denominator = self.num_words_assigned_to_topic.astype(np.float32)

            # Exclude current word

            numerator[old_topic] -= 1.0
            denominator[old_topic] -= 1.0

            # Smoothing

            numerator += self.beta
            denominator += self.vocab_size * self.beta

            # Apply multiplier

            proportions *= numerator
            proportions /= denominator

        # Draw topic

        new_topic = np.random.multinomial(1, proportions / np.sum(proportions))[0]
        logging.info("Drew topic = %d", new_topic)
        self.topic_assignments[document_idx][sentence_idx][word_idx] = new_topic

        # Correct counts

        self.num_words_in_doc_assigned_to_topic[document_idx][old_topic] -= 1 
        self.num_words_in_doc_assigned_to_topic[document_idx][new_topic] += 1

        self.num_same_words_assigned_to_topic[word][old_topic] -= 1
        self.num_same_words_assigned_to_topic[word][new_topic] += 1

        self.num_words_assigned_to_topic[old_topic] -= 1
        self.num_words_assigned_to_topic[new_topic] += 1


    def draw_class(self, document_idx, sentence_idx, word_idx):
        old_class = self.class_assignments[document_idx][sentence_idx][word_idx]
        word = self.documents[document_idx][sentence_idx][word_idx]

        # Get neighboring classes

        if word_idx > 0:
            previous = self.class_assignments[document_idx][sentence_idx][word_idx - 1]
        else:
            previous = None

        try:
            future = self.class_assignments[document_idx][sentence_idx][word_idx + 1]
        except IndexError:
            future = None

        # Build first term of numerator

        if previous is not None:
            term_1 = self.num_transitions[previous, :].astype(np.float32)
        else:
            term_1 = np.zeros(self.num_classes, dtype=np.float32)
        term_1 += self.gamma

        # Build second term of numerator
        
        if future is not None:
            term_2 = self.num_transitions[:, future].astype(np.float32)
        else:
            term_2 = np.zeros(self.num_classes, dtype=np.float32)
        if previous is not None and future is not None and previous == future:
            term_2[previous] += 1.0
        term_2 += self.gamma

        # Calculate numerator

        numerator = term_1 * term_2

        # Build denominator

        denominator = self.num_words_assigned_to_class
        if previous is not None:
            denominator[previous] += 1.0
        denominator += self.num_classes * self.gamma

        # Calculate multiplier

        if self.class_assignments[document_idx][sentence_idx][word_idx] == 0:

            # Initialize numerator of multiplier with same word topic counts

            multiplier = self.num_same_words_assigned_to_topic[word].astype(np.float32)

            # Smoothing

            multiplier += self.beta

            # Divide by denominator of multiplier

            multiplier /= self.num_words_assigned_to_topic.astype(np.float32) + self.vocab_size * self.beta

        else:

            # Initialize numerator of multiplier with same word class counts

            multiplier = self.num_same_words_assigned_to_class[word].astype(np.float32)

            # Smoothing

            multiplier += self.delta

            # Divide by denominator of multiplier

            multiplier /= self.num_words_assigned_to_class + self.vocab_size * self.delta

        # Calculate probability proportions

        proportions = multiplier * numerator / denominator

        # Draw class

        new_class = np.random.multinomial(1, proportions / np.sum(proportions))[0]
        logging.info("Drew class = %d", new_class)
        self.class_assignments[document_idx][sentence_idx][word_idx] = new_class

        # Correct counts

        self.num_transitions[previous, old_class] -= 1
        self.num_transitions[previous, new_class] += 1

        self.num_transitions[old_class, future] -= 1
        self.num_transitions[new_class, future] += 1

        self.num_same_words_assigned_to_class[word][old_class] -= 1
        self.num_same_words_assigned_to_class[word][new_class] += 1

        self.num_words_assigned_to_class[old_class] -= 1
        self.num_words_assigned_to_class[new_class] += 1
