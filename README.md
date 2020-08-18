# Tree-to-sequence autoencoder

Autoencoder architecture designed to convert dependency parse trees back into their originating sentences. This process recovers the word order in the original text, which is lost in the dependency parse tree. This model includes an additional word embedding layer, since its main purpose is to learn structure-aware word representations.

Our model is based on the idea of sequence-to-sequence encoding ([Sutskever et al. 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)), for which it takes part of the structure of the code from [Ben Trevett's tutorial](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb).

The tree encoder is adapted from the [Unbounce PyTorch implementation](https://github.com/unbounce/pytorch-tree-lstm) of the Tree-LSTM child-sum model from ([Tai et al. 2015](https://arxiv.org/abs/1503.00075)).



## Dataset

## Usage

