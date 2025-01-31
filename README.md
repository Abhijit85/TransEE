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


   mermaid
graph TD
    A[Input Triplet (h, r, t)] --> B[Retrieve Embeddings (e_h, e_r, e_t)]
    B --> C[Split Embeddings into Modulus and Phase]
    C --> D[Ensure Phase in [-π, π]]
    D --> E[Adjust Relation Modulus with Bias]
    E --> F[Transformation: Combine Modulus and Phase]
    F --> G[Compute Phase Score]
    F --> H[Compute Modulus Score]
    G --> J[Scoring Function (L2 Norm)]
    H --> J
    J --> I[Calculate Final Score (Phase + Modulus)]
    I --> K[Apply Regularization (Phase and Modulus)]
    K --> L[Output: Triplet Plausibility Score]

    %% Step details
    subgraph Details
        C1[Split: Extract modulus (m) and phase (φ) from embeddings. Formula: e_h = (m_h, φ_h), e_r = (m_r, φ_r), e_t = (m_t, φ_t). Modulus captures magnitude, and phase captures angular information.]
        C2[Phase Restriction: Normalize phase values to fall within [-π, π]. Formula: φ = φ mod (2π).]
        C3[Modulus Bias: Adjust relation modulus by clamping and applying bias to ensure stability. Formula: m_r = max(0, min(m_r, 1)) + bias. Handle irregularities by constraining values.]
        C4[Combine: Merge adjusted modulus and phase to create transformed embeddings. Formula: e_h + e_r - e_t = (m_h \* m_r, φ_h + φ_r - φ_t).]
        C5[Phase Score: Compute alignment between head, relation, and tail phases. Formula: Phase Score = |sin((φ_h + φ_r - φ_t) / 2)|.]
        C6[Modulus Score: Calculate Euclidean distance between scaled modulus values of head, relation, and tail. Formula: Modulus Score = ||m_h \* m_r - m_t||_2.]
        C7[L2 Norm: Aggregate phase and modulus scores using L2 norm. Formula: Score = γ - (Phase Score + Modulus Score).]
        C8[Regularization: Apply phase and modulus-specific constraints to embeddings to prevent overfitting. Formula: Regularization = L3(e_h) + L3(e_r).]
    end

    %% Connect details
    C --> C1
    D --> C2
    E --> C3
    F --> C4
    G --> C5
    H --> C6
    J --> C7
    K --> C8


```


























## Acknowledgement
We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). Thanks for their contributions.

## Other Repositories
If you are interested in our work, you may find the following paper useful.
