# TransEE-Enhancing-Knowledge-Graph-Embedding-for-Complex-Relations

## Introduction
This is a PyTorch implementation of TransEE for learning embeddings in knowledge graphs (KGE). TransEE introduces enhancements that improve handling of complex relations (one-to-many, many-to-one, many-to-many) in knowledge graphs. The implementation is optimized for fast training on large-scale knowledge graphs and can achieve state-of-the-art performance on datasets like FB15K, WN18, and YAGO3-10.

## Implemented Features
-:Supports diverse relational patterns
-:Enhanced Relation Expressiveness
-:Improved Scoring Mechanism
-:Efficient Training & Inference


## Evaluation Metrics
We evaluate TransEE using the following standard KGE metrics:
-Mean Rank (MR): Measures the average rank of correct entities
-Hits@10 (Filtered): Percentage of test triples where the correct entity is ranked in the top 10 after filtering out corrupted triples



## Loss Function:
-Uniform Negative Sampling
-Self-Adversarial Negative Sampling

## Supported Datasets

add table with datasets - entities, relations...

## Usage

Knowledge Graph Data Format:

The dataset consists of the following files:

entities.dict – A dictionary mapping entities to unique IDs
relations.dict – A dictionary mapping relations to unique IDs
train.txt – The dataset used to train the KGE model
valid.txt – The validation dataset 
test.txt – The dataset used to evaluate the KGE model


# Testing
python run.py --do_test --data_path ./data/FB15K --init_checkpoint ./output/TransEEnhanced_FB15K --test_batch_size 16 --cuda

## Results of TransEE model
add results with all datasets

# Knowledge Graph Embedding Repository Structure

This repository is designed for training and evaluating Knowledge Graph Embedding (KGE) models such as TransE, RotatE, and TransEEnhanced on datasets like FB15K.

## Directory Structure

```plaintext
knowledge_graph_embedding/
├── data/                  # Directory for datasets
│   ├── FB15K/             # Dataset folder (e.g., FB15K)
│   │   ├── train.txt      # Training triples
│   │   ├── valid.txt      # Validation triples
│   │   ├── test.txt       # Testing triples
│   │   ├── entities.dict  # Entity-to-ID mapping
│   │   ├── relations.dict # Relation-to-ID mapping
├── output/                # Directory to store trained models and results
│   ├── TransEEnhanced_FB15K/  # Model checkpoint for TransEEnhanced
│   │   ├── config.json        # Saved configuration
│   │   ├── checkpoint         # Model and optimizer state
│   │   ├── entity_embedding.npy   # Saved entity embeddings
│   │   ├── relation_embedding.npy # Saved relation embeddings
├── dataloader.py          # Contains dataset classes (TrainDataset, TestDataset, etc.)
├── model.py               # Implements KGEModel and scoring functions
├── run.py                 # Main script for training, validation, and testing
├── utils.py               # Utility functions (e.g., categorize relations, filter test triples)
├── requirements.txt       # Python dependencies
├── README.md              # Project description and instructions

```

## Citation

add citations for RotateE , thank them
























## Acknowledgement
We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). Thanks for their contributions.

## Other Repositories
If you are interested in our work, you may find the following paper useful.
