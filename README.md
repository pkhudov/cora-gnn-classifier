# cora-gnn-classifier

## Exercise to develop familiarity with PyTorch Geometric and GNN training

The goal of this notebook is to develop first experience in training Graph Neural Networks using Pytorch Geometric, by performing node classification on a Cora dataset. It consists of a citation network of small size, where nodes represent documents, edges the citations, and embeddings some of the words used in the document. The task is to infer the category of each document, out of 7. It will be an instance of semi-supervised learning, more concretely transductive learning, where only some of the nodes are labelled, thus requiring us to classify the rest.

The exercise will focus on intuitively tuning the hyperparameters of GCN (Graph Convolutional Neural Network) and GAT (Graph Attention Neural Network). As there is potentially an unlimited amount of ways in which it can be done, the starting point is an already developed model from a tutorial on Pytorch Geometric and a paper by Velickovic et al., for GCN and GAT respectively. Another reason for not starting from zero is that I read these resources before starting on each section, which means that anyway I would be unable to make unbiased decisions.

The more specific objective for each model tuning is outlined in the corresponding model section.
