# TransEE-Enhancing-Knowledge-Graph-Embedding-for-Complex-Relations


# Testing
python run.py --do_test --data_path ./data/FB15K --init_checkpoint ./output/TransEEnhanced_FB15K --test_batch_size 16 --cuda



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


























## Acknowledgement
We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). Thanks for their contributions.

## Other Repositories
If you are interested in our work, you may find the following paper useful.
