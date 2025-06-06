ConceptUML: Multiphase Unsupervised Threat Detection Via Concept Learning, Hidden Markov Models and Topic Modelling
----------------------------------------------------------

This code implements an end-to-end pipeline to analyze event log sequences using:
- BERT sentence embeddings to generate contextualized representations of event sequences.
- Non-negative Matrix Factorization (NMF) for concept learning on BERT embeddings.
- Hidden Markov Models (HMM) for clustering event sequences, selecting the best number of hidden states via log-likelihood maximization.
- Cosine similarity computation with external cybersecurity knowledge bases (MITRE ATT&CK, CAPEC).
- Identification of the most suspicious cluster based on combined MITRE and CAPEC similarity scores.
- Output of results to Excel files:
    - Full results with cluster assignments.
    - Cluster-level similarity scores.
    - Suspicious cluster ranking by similarity.
    - Runtime and memory usage statistics.

Key Features:
-------------
- BERT Embedding: SentenceTransformer ('bert-base-nli-mean-tokens')
- NMF Concept Learning on BERT embeddings with configurable number of concepts.
- HMM model selection across a range of hidden states
- Custom rule-based tokenization for command-line strings.
- Cosine similarity evaluation with MITRE and CAPEC token datasets.

Author: Dr. Khanh Luong
Last Updated: June 2025

Intended Usage:
---------------
This code is intended for academic publication and reproducibility. 
Feel free to cite or reuse parts with proper attribution.

Dependencies:
-------------
- pandas, numpy, scikit-learn, hmmlearn
- sentence-transformers, transformers
- gensim, dask, scipy
- psutil, openpyxl
"""
