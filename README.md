# Replication of Neural Graph Fingerprints


This software package replicates the work by David Duvenaud, *et. al.*, on the implementation of convolutional networks on molecular graphs of arbitrary size that generate molecular fingerprints (i.e., embeddings) for downstream property prediction. Citation to the original paper:

[Duvenaud, D. K., Maclaurin, D., Iparraguirre, J., Bombarell, R., Hirzel, T., Aspuru-Guzik, A., and Adams, R. P. (2015) Convolutional networks on graphs for learning molecular fingerprints. *Advances in Neural Information Processing Systems*, 2224-2232.](http://arxiv.org/pdf/1509.09292.pdf)

This replication work also uses part of the original authors' preprocessed data and code, which can be found in their [Github repository](https://github.com/HIPS/neural-fingerprint). All the data needed for this replication are available in this repository under the `data` folder.


## Package setup

This package requires:
* [Numpy version >= 1.26.0](https://numpy.org/)
* [Pandas version >= 2.2.2](https://pandas.pydata.org/)
* [PyTorch version >= 2.1.2](https://pytorch.org/)
* [Scikit-learn version >= 1.4.0](https://scikit-learn.org/stable/) 
* [Matplotlib version >= 3.8.2](https://matplotlib.org/)
* [RDkit version >= 2023.9.5](https://www.rdkit.org/)

