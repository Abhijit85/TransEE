# TransEE-Enhancing-Knowledge-Graph-Embedding-for-Complex-Relations


# Testing
python run.py --do_test --data_path ./data/FB15K --init_checkpoint ./output/TransEEnhanced_FB15K --test_batch_size 16 --cuda



###########################################

knowledge_graph_embedding/
├── data/                  # Directory for dataset files
│   ├── FB15K/             # Dataset folder (FB15K)
│   │   ├── train.txt      # Training triples
│   │   ├── valid.txt      # Validation triples
│   │   ├── test.txt       # Testing triples
│   │   ├── entities.dict  # Entity to ID mapping
│   │   ├── relations.dict # Relation to ID mapping
│   │   ├── regions.list   # (Optional) Region file for specific datasets
├── dataloader.py          # Contains TrainDataset, TestDataset, etc.
├── model.py               # Contains KGEModel and scoring functions
├── run.py                 # Main script for training and evaluation
├── utils.py               # Utility functions (e.g., categorize relations, filter triples)
├── README.md              # Project description and instructions
#############################################




























## Acknowledgement
We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). Thanks for their contributions.

## Other Repositories
If you are interested in our work, you may find the following paper useful.
