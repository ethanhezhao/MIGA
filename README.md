# MIGA

MIGA is a short text clustering/aggregation topic model that leverages document metadata, which enjoys improved performance on short text topic modelling and clustering. MIGA is able to cluster documents according to both their semantics (captured by topics) and their metadata. Each cluster can also be visualised by their topics.

MIGA is detailed in the paper "Leveraging Meta Information in Short Text Aggregation" in ACL 2019 [Link](https://www.aclweb.org/anthology/P19-1396/).

# Run MIGA

1. Requirements: Matlab 2016b (or later).

3. We have offered the WS dataset used in the paper, which is stored in MAT format, with the following contents:
- doc: A V by N sparse matrix of the documents, where V and N are the number of unique words and the number of documents, respectively.
- doc_label: the one-hot representation of the document labels.
- voc: vocabulary of the dataset.
- label_name: the name of the document labels.

Please prepare your own documents in the above format. If you want to use this dataset, please cite the original papers, which are cited in our paper.

4. Run ```demo_MIGA.m```.

5. Important parameters:
- Para.M: the number of clusters.
- Para.K: the number of topics.
- Para.is_meta_data: whether document metadata is used to aggregate/cluster documents.
- Para.is_sample_alpha: whether the prior, i.e., alpha is sampled

6. Outputs:

The code saves the necessary statistics of the model, in a MAT file named ```./save/model.mat```.

The code also prints the most important clusters for each document label, the most important documents and topics for each cluster, as shown in the paper.
The visualisation can be useful for detailed understanding of the input corpus.
