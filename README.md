Hidden Markov Model Latent Dirichlet Allocation in Python

This is a Python implementation of HMM-LDA as presented in T.L. Griffiths et al. in "Integrating Topics and Syntax". It uses Gibbs sampling for inference as per section 2.2. It is a work in progress.

# Goals

Provide a readable, annotated codebase for future work on HMM-LDA and variations.

Test the relative effectiveness of different initaliza

# Installation

This package was developed with Python 2.7. pip is the recommended package manager, though NumPy is the only dependency. Dependencies can be installed using `pip install -r requirements.txt`

# Testing

The test suite, which is still a work in progress can be run using the command `./test.sh` which runs Python's unittest framework

# Experiments

The numpy_hmm_lda.experiments module contains the generate module

## Generate

`python -m numpy_hmm_lda.experiments.generate`

Generate is the following experiment, given a few settings, including hyperparameters, draw document topic distributions (theta) topic/class word distributions (phi) and transition matrix rows (pi). From these, generate documents according to the model. Feed these documents back into the model and perform inference. We expect to see a permutation of the original distributions after sufficient training.

Generate outputs snapshots of the sampler as .dat files (formatted as tab-separated values)

These .dat files can be plotted as .png images using `./plot.sh`. Note that this functionality depends on gnuplot (http://www.gnuplot.info/)
