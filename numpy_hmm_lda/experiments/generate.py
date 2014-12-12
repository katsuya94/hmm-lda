'''
Start from hyperparameters and generate documents.
Generating distributions and inferred distributions should be equal, ubject to permutations.
'''

import numpy as np
from ..model import HiddenMarkovModelLatentDirichletAllocation

num_topics = 4
num_classes = 4
num_documents = 16
num_words = 64
num_words_per_document = 256

alpha_scalar = 0.5
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

def generate(document_idx):
    document = np.empty(num_words_per_document, dtype=np.int64)
    topic_assignments = np.empty(num_words_per_document, dtype=np.int64)
    class_assignments = np.empty(num_words_per_document, dtype=np.int64)
    prev_class = None
    for word_idx in range(num_words_per_document):
        topic_draw = np.random.multinomial(1, per_document_topic_distributions[document_idx])[0]
        if prev_class is not None:
            class_draw = np.random.multinomial(1, transition_matrix[prev_class])[0]
        else:
            class_draw = np.random.randint(num_classes)
        if class_draw == 0:
            word_draw = np.random.multinomial(1, per_topic_word_distributions[topic_draw])[0]
        else:
            word_draw = np.random.multinomial(1, per_class_word_distributions[class_draw])[0]
        topic_assignments[word_idx] = topic_draw
        class_assignments[word_idx] = class_draw
        document[word_idx] = word_draw
        prev_class = class_draw
    return document, topic_assignments, class_assignments

model = HiddenMarkovModelLatentDirichletAllocation(vocab_size=num_words,
                                                   num_topics=num_topics,
                                                   num_classes=num_classes,
                                                   alpha=alpha_scalar,
                                                   beta=beta_scalar,
                                                   gamma=gamma_scalar,
                                                   delta=delta_scalar)

for document_idx in range(num_documents):
    document, topic_assignments, class_assignments = generate(document_idx)
    model.add_document(document)
    # model.add_document_with_initial_configuration(document, topic_assignments, class_assignments)

# model.run_counts()
model.initialize()

per_document_topic_distributions_over_time = [[] for i in range(num_documents)]
class_totals_over_time = []
topic_totals_over_time = []

for i in range(10):
    model.train(1)
    for idx, topic_distribution in enumerate(model.per_document_topic_distributions()):
        per_document_topic_distributions_over_time[idx].append(topic_distribution)
    class_totals_over_time.append(model.num_words_assigned_to_class.copy())
    topic_totals_over_time.append(model.num_words_assigned_to_topic.copy())

for idx, topic_distribution in enumerate(model.per_document_topic_distributions()):
    print '%d: %s %s' % (idx, topic_distribution, per_document_topic_distributions[idx])

for idx, topic_distributions_over_time in enumerate(per_document_topic_distributions_over_time):
    np.savetxt('document%d.dat' % idx, np.array(topic_distributions_over_time), delimiter='\t')

np.savetxt('classtotals.dat', np.array(class_totals_over_time), delimiter='\t')
np.savetxt('topictotals.dat', np.array(topic_totals_over_time), delimiter='\t')
