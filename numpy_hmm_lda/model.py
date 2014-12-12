import logging
import numpy as np

def categorical(proportions):
    draw = np.random.uniform(0, np.sum(proportions))
    for idx, cumsum in enumerate(np.cumsum(proportions)):
        if draw < cumsum:
            return idx
    return proportions.size - 1

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
        self.topic_assignments = []
        self.class_assignments = []

    def set_vocab_size(self, size):
        '''
        Sets the size of the vocabulary.
        size = integer vocabulary size
        '''
        self.vocab_size = size

    def add_document(self, document):
        '''
        Adds a document to the model.
        document = numpy array of integers representing words
        '''
        logging.info('Adding document of %d words', document.size)
        self.documents.append(document)

    def add_document_with_initial_configuration(self, document, topic_assignments, class_assignments):
        '''
        Adds a document to the model along with a configuration.
        No need to initialize when doing this only to run_counts.
        document = numpy array of integers representing words
        topic_assignments = numpy array of integers representing topics
        class_assignments = numpy array of integers representing classes
        '''
        logging.info('Adding document of %d words', document.size)
        self.documents.append(document)
        self.topic_assignments.append(topic_assignments)
        self.class_assignments.append(class_assignments)

    def initialize(self):
        '''
        Set random assignments of topics and classes initially.
        '''
        self.topic_assignments = [np.random.randint(0, self.num_topics, size=document.size) for document in self.documents]
        self.class_assignments = [np.random.randint(0, self.num_classes, size=document.size) for document in self.documents]
        self.run_counts()

    def run_counts(self):
        '''
        Run all counting routines based on current configuration.
        '''
        self.num_words_in_doc_assigned_to_topic = [self.count_num_words_in_doc_assigned_to_topic(document_idx) for document_idx in xrange(len(self.documents))]
        self.num_same_words_assigned_to_topic = [self.count_num_same_words_assigned_to_topic(word) for word in xrange(self.vocab_size)]
        self.num_words_assigned_to_topic = self.count_num_words_assigned_to_topic()
        self.num_same_words_assigned_to_class = [self.count_num_same_words_assigned_to_class(word) for word in xrange(self.vocab_size)]
        self.num_words_assigned_to_class = self.count_num_words_assigned_to_class()
        self.num_transitions = self.count_num_transitions()

    def initialize_progressive(self):
        '''
        Start with empty counts.
        Use the counts to progressively draw the next word.
        '''
        self.topic_assignments = [np.empty(document.size) for document in self.documents]
        self.class_assignments = [np.empty(document.size) for document in self.documents]
        self.num_words_in_doc_assigned_to_topic = [np.zeros(self.num_topics) for document in self.documents]
        self.num_same_words_assigned_to_topic = [np.zeros(self.num_topics) for word in xrange(self.vocab_size)]
        self.num_words_assigned_to_topic = np.zeros(self.num_topics)
        self.num_same_words_assigned_to_class = [np.zeros(self.num_classes) for word in xrange(self.vocab_size)]
        self.num_words_assigned_to_class = np.zeros(self.num_classes)
        self.num_transitions = np.zeros((self.num_classes, self.num_classes,))

        for document_idx in xrange(len(self.documents)):
            prev_class = None
            for word_idx in xrange(self.documents[document_idx].size):
                self.draw_class_uncounted(document_idx, word_idx)
                self.draw_topic_uncounted(document_idx, word_idx)

    def check(self):
        '''
        Print inconsistencies in the current counts.
        '''
        for test, check in zip(self.num_words_in_doc_assigned_to_topic, [self.count_num_words_in_doc_assigned_to_topic(document_idx) for document_idx in xrange(len(self.documents))]):
            try:
                assert (test == check).all()
            except AssertionError:
                print 'num_words_in_doc_assigned_to_topic', test, check
        for test, check in zip(self.num_same_words_assigned_to_topic, [self.count_num_same_words_assigned_to_topic(word) for word in xrange(self.vocab_size)]):
            try:
                assert (test == check).all()
            except AssertionError:
                print 'num_same_words_assigned_to_topic', test, check
        try:
            assert (self.num_words_assigned_to_topic == self.count_num_words_assigned_to_topic()).all()
        except AssertionError:
            print 'num_words_assigned_to_topic', self.num_words_assigned_to_topic, self.count_num_words_assigned_to_topic()
        for test, check in zip(self.num_same_words_assigned_to_class, [self.count_num_same_words_assigned_to_class(word) for word in xrange(self.vocab_size)]):
            try:
                assert (test == check).all()
            except AssertionError:
                print 'num_same_words_assigned_to_class', test, check
        try:
            assert (self.num_words_assigned_to_class == self.count_num_words_assigned_to_class()).all()
        except AssertionError:
            print 'num_words_assigned_to_class', self.num_words_assigned_to_class, self.count_num_words_assigned_to_class()
        try:
            assert (self.num_transitions == self.count_num_transitions()).all()
        except AssertionError:
            print 'num_transitions', self.num_transitions, self.count_num_transitions()
        try:
            assert np.sum(self.num_words_assigned_to_class) == sum(document.size for document in self.documents)
        except:
            print 'num_words_assigned_to_class sum', np.sum(self.num_words_assigned_to_class), sum(document.size for document in self.documents)
        try:
            assert np.sum(self.num_transitions) == sum(document.size - 1 for document in self.documents)
        except:
            print 'num_transitions sum', np.sum(self.num_transitions), sum(document.size - 1 for document in self.documents)

    def count_num_words_in_doc_assigned_to_topic(self, document_idx):
        topic_counts = np.zeros(self.num_topics, dtype=np.int64)
        for word_idx in xrange(self.documents[document_idx].size):
            if self.class_assignments[document_idx][word_idx] == 0:
                topic_counts[self.topic_assignments[document_idx][word_idx]] += 1
        return topic_counts

    def count_num_same_words_assigned_to_topic(self, word):
        topic_counts = np.zeros(self.num_topics, dtype=np.int64)
        for document_idx in xrange(len(self.documents)):
            for word_idx in xrange(self.documents[document_idx].size):
                if self.documents[document_idx][word_idx] == word and self.class_assignments[document_idx][word_idx] == 0:
                    topic_counts[self.topic_assignments[document_idx][word_idx]] += 1
        return topic_counts

    def count_num_words_assigned_to_topic(self):
        topic_counts = np.zeros(self.num_topics, dtype=np.int64)
        for document_idx in xrange(len(self.documents)):
            for word_idx in xrange(self.documents[document_idx].size):
                if self.class_assignments[document_idx][word_idx] == 0:
                    topic_counts[self.topic_assignments[document_idx][word_idx]] += 1
        return topic_counts

    def count_num_same_words_assigned_to_class(self, word):
        class_counts = np.zeros(self.num_classes, dtype=np.int64)
        for document_idx in xrange(len(self.documents)):
            for word_idx in xrange(self.documents[document_idx].size):
                if self.documents[document_idx][word_idx] == word:
                    class_counts[self.class_assignments[document_idx][word_idx]] += 1
        return class_counts

    def count_num_words_assigned_to_class(self):
        class_counts = np.zeros(self.num_classes, dtype=np.int64)
        for document_idx in xrange(len(self.documents)):
            for word_idx in xrange(self.documents[document_idx].size):
                class_counts[self.class_assignments[document_idx][word_idx]] += 1
        return class_counts

    def count_num_transitions(self):
        transition_counts = np.zeros((self.num_classes, self.num_classes,), dtype=np.int64)
        for document_idx in xrange(len(self.documents)):
            previous = None
            for word_idx in xrange(self.documents[document_idx].size):
                current = self.class_assignments[document_idx][word_idx]
                if previous is not None:
                    transition_counts[previous, current] += 1
                previous = current
        return transition_counts

    def train(self, iterations):
        '''
        Run Gibbs sampling
        iterations = number of full samplings of the corpus
        '''
        tenth = max(iterations // 10, 1)
        for i in xrange(iterations):
            if i % tenth == 0:
                print 'iteration %d' % i
            for document_idx in xrange(len(self.documents)):
                for word_idx in xrange(self.documents[document_idx].size):
                    logging.info('[%d][%d] = %d', document_idx, word_idx, self.documents[document_idx][word_idx])
                    self.draw_class(document_idx, word_idx)
                    self.draw_topic(document_idx, word_idx)

    def draw_topic_uncounted(self, document_idx, word_idx):
        old_class = self.class_assignments[document_idx][word_idx]
        word = self.documents[document_idx][word_idx]

        # Initialize probability proportions with document topic counts

        proportions = self.num_words_in_doc_assigned_to_topic[document_idx].astype(np.float64)

        # Smoothing

        proportions += self.alpha

        # If the current word is assigned to the semantic class

        if old_class == 0:

            # Initialize numerator with same word topic counts

            numerator = self.num_same_words_assigned_to_topic[word].astype(np.float64)

            # Initialize denominator with global topic counts

            denominator = self.num_words_assigned_to_topic.astype(np.float64)

            # Smoothing

            numerator += self.beta
            denominator += self.vocab_size * self.beta

            # Apply multiplier

            proportions *= numerator
            proportions /= denominator

        # Draw topic

        logging.info('proportions = %s', proportions)
        new_topic = categorical(proportions)
        logging.info('drew topic %d', new_topic)
        self.topic_assignments[document_idx][word_idx] = new_topic

        # Correct counts

        if old_class == 0:
            self.num_words_in_doc_assigned_to_topic[document_idx][new_topic] += 1
            self.num_same_words_assigned_to_topic[word][new_topic] += 1
            self.num_words_assigned_to_topic[new_topic] += 1

    def draw_topic(self, document_idx, word_idx):
        old_topic = self.topic_assignments[document_idx][word_idx]
        old_class = self.class_assignments[document_idx][word_idx]
        word = self.documents[document_idx][word_idx]

        # Initialize probability proportions with document topic counts

        proportions = self.num_words_in_doc_assigned_to_topic[document_idx].astype(np.float64)

        # Exclude current word

        if old_class == 0:
            proportions[old_topic] -= 1.0

        # Smoothing

        proportions += self.alpha

        # If the current word is assigned to the semantic class

        if old_class == 0:

            # Initialize numerator with same word topic counts

            numerator = self.num_same_words_assigned_to_topic[word].astype(np.float64)

            # Initialize denominator with global topic counts

            denominator = self.num_words_assigned_to_topic.astype(np.float64)

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

        logging.info('proportions = %s', proportions)
        new_topic = categorical(proportions)
        logging.info('drew topic %d', new_topic)
        self.topic_assignments[document_idx][word_idx] = new_topic

        # Correct counts

        if old_class == 0:
            self.num_words_in_doc_assigned_to_topic[document_idx][old_topic] -= 1 
            self.num_words_in_doc_assigned_to_topic[document_idx][new_topic] += 1

            self.num_same_words_assigned_to_topic[word][old_topic] -= 1
            self.num_same_words_assigned_to_topic[word][new_topic] += 1

            self.num_words_assigned_to_topic[old_topic] -= 1
            self.num_words_assigned_to_topic[new_topic] += 1

    def draw_class_uncounted(self, document_idx, word_idx):
        word = self.documents[document_idx][word_idx]

        # Get neighboring classes

        if word_idx > 0:
            previous = self.class_assignments[document_idx][word_idx - 1]
        else:
            previous = None

        # Build first term of numerator

        if previous is not None:
            term_1 = self.num_transitions[previous, :].astype(np.float64)
        else:
            term_1 = np.zeros(self.num_classes, dtype=np.float64)
        term_1 += self.gamma

        # Build second term of numerator
        
        term_2 = np.full(self.num_classes, self.gamma, dtype=np.float64)

        # Calculate numerator

        numerator = term_1 * term_2

        # Build denominator

        denominator = self.num_words_assigned_to_class.astype(np.float64)
        if previous is not None:
            denominator[previous] += 1.0
        denominator += self.num_classes * self.gamma

        # Calculate multiplier

        multiplier = np.empty(self.num_classes, dtype=np.float64)

        # Initialize numerator of multiplier with same word topic counts

        multiplier = self.num_same_words_assigned_to_class[word].astype(np.float64)

        # Smoothing

        multiplier += self.delta

        # Divide by denominator of multiplier

        multiplier /= self.num_words_assigned_to_class.astype(np.float64) + self.vocab_size * self.delta

        # Calculate probability proportions

        proportions = multiplier * numerator / denominator

        # Draw class

        logging.info('proportions = %s', proportions)
        new_class = categorical(proportions)
        logging.info('drew class %d', new_class)
        self.class_assignments[document_idx][word_idx] = new_class

        # Correct counts

        if previous is not None:
            self.num_transitions[previous, new_class] += 1
        self.num_same_words_assigned_to_class[word][new_class] += 1
        self.num_words_assigned_to_class[new_class] += 1

    def draw_class(self, document_idx, word_idx):
        old_class = self.class_assignments[document_idx][word_idx]
        old_topic = self.topic_assignments[document_idx][word_idx]
        word = self.documents[document_idx][word_idx]

        # Get neighboring classes

        if word_idx > 0:
            previous = self.class_assignments[document_idx][word_idx - 1]
        else:
            previous = None

        try:
            future = self.class_assignments[document_idx][word_idx + 1]
        except IndexError:
            future = None

        # Build first term of numerator

        if previous is not None:
            term_1 = self.num_transitions[previous, :].astype(np.float64)
            term_1[old_class] -= 1.0 # Exclude current word
        else:
            term_1 = np.zeros(self.num_classes, dtype=np.float64)
        term_1 += self.gamma

        # Build second term of numerator
        
        if future is not None:
            term_2 = self.num_transitions[:, future].astype(np.float64)
            term_2[old_class] -= 1.0 # Exclude current word
        else:
            term_2 = np.zeros(self.num_classes, dtype=np.float64)
        if previous is not None and future is not None and previous == future:
            term_2[previous] += 1.0
        term_2 += self.gamma

        # Calculate numerator

        numerator = term_1 * term_2

        # Build denominator

        denominator = self.num_words_assigned_to_class.astype(np.float64)
        if previous is not None:
            denominator[previous] += 1.0
        denominator += self.num_classes * self.gamma

        # Calculate multiplier

        multiplier_numerator = np.empty(self.num_classes, dtype=np.float64)
        multiplier_denominator = np.empty(self.num_classes, dtype=np.float64)

        # Initialize numerator of multiplier with same word class/topic counts

        multiplier_numerator[1:] = self.num_same_words_assigned_to_class[word][1:].astype(np.float64)
        multiplier_numerator[0] = self.num_same_words_assigned_to_topic[word][old_topic].astype(np.float64)

        # Exclude current word

        if old_class != 0:
            multiplier_numerator[old_class] -= 1.0
        multiplier_numerator[0] -= 1.0

        # Smoothing

        multiplier_numerator[1:] += self.delta
        multiplier_numerator[0] += self.beta

        # Initialize denominator of multiplier with global class/topic counts

        multiplier_denominator[1:] = self.num_words_assigned_to_class[1:].astype(np.float64)
        multiplier_denominator[0] = self.num_words_assigned_to_topic[old_topic].astype(np.float64)

        # Exclude current word

        if old_class != 0:
            multiplier_denominator[old_class] -= 1.0
        multiplier_denominator[0] -= 1.0

        # Smoothing

        multiplier_denominator[1:] += self.delta * self.vocab_size
        multiplier_denominator[0] += self.beta * self.vocab_size

        # Calculate probability proportions

        proportions = (multiplier_numerator / multiplier_denominator) * numerator / denominator

        # Draw class

        logging.info('proportions = %s', proportions)
        new_class = categorical(proportions)
        logging.info('drew class %d', new_class)
        self.class_assignments[document_idx][word_idx] = new_class

        # Correct counts

        if previous is not None:
            self.num_transitions[previous, old_class] -= 1
            self.num_transitions[previous, new_class] += 1

        if future is not None:
            self.num_transitions[old_class, future] -= 1
            self.num_transitions[new_class, future] += 1

        self.num_same_words_assigned_to_class[word][old_class] -= 1
        self.num_same_words_assigned_to_class[word][new_class] += 1

        self.num_words_assigned_to_class[old_class] -= 1
        self.num_words_assigned_to_class[new_class] += 1

        if old_class == 0 and new_class != 0:
            self.num_words_in_doc_assigned_to_topic[document_idx][old_topic] -= 1
            self.num_same_words_assigned_to_topic[word][old_topic] -= 1
            self.num_words_assigned_to_topic[old_topic] -= 1
        elif old_class != 0 and new_class == 0:
            self.num_words_in_doc_assigned_to_topic[document_idx][old_topic] += 1
            self.num_same_words_assigned_to_topic[word][old_topic] += 1
            self.num_words_assigned_to_topic[old_topic] += 1

    def per_document_topic_distributions(self):
        def count(document_idx):
            topic_counts = np.zeros(self.num_topics, dtype=np.int64)
            for word_idx in xrange(self.documents[document_idx].size):
                topic_counts[self.topic_assignments[document_idx][word_idx]] += 1
            return topic_counts
        per_document_topic_counts = [count(document_idx) for document_idx in xrange(len(self.documents))]
        return [topic_counts.astype(np.float64) / np.sum(topic_counts) for topic_counts in per_document_topic_counts]
