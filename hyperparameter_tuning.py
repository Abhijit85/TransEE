from dotenv import load_dotenv
import torch
import itertools
from data_loader import custom_dataset
from model import TransEEnhanced
from traning import training


# Load dataset using CustomDataset class
data_path = '/content/sample_data'
dataset = CustomDataset(data_path)
data = dataset

# Extract edge indices and types
train_edge_index, train_edge_type = dataset.get_edge_indices_and_types(dataset.train_data)
valid_edge_index, valid_edge_type = dataset.get_edge_indices_and_types(dataset.valid_data)
test_edge_index, test_edge_type = dataset.get_edge_indices_and_types(dataset.test_data)

# Define the hyperparameter grid
param_grid = {
    "embedding_dim": [128, 256, 512],
    "learning_rate": [0.001, 0.005, 0.01],
    "margin": [5.0, 9.0, 12.0],
    "batch_size": [512, 1024],
    "phase_weight": [0.5, 1.0],
    "modulus_weight": [1.0, 2.0]
}

# Generate all combinations of hyperparameters
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Placeholder for the best configuration and its performance
best_config = None
best_mrr = -float('inf')

# Iterate through each combination of hyperparameters
for config in param_combinations:
    print(f"Testing configuration: {config}")

    # Initialize the model with the current hyperparameter configuration
    model = TransEEnhanced(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        embedding_dim=config['embedding_dim'],
        margin=config['margin'],
        phase_weight=config['phase_weight'],
        modulus_weight=config['modulus_weight']
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.99))

    # Create a data object with the extracted edge indices and types
    data.train_edge_index = train_edge_index
    data.train_edge_type = train_edge_type
    data.valid_edge_index = valid_edge_index
    data.valid_edge_type = valid_edge_type
    data.test_edge_index = test_edge_index
    data.test_edge_type = test_edge_type

    # Train the model
    train(
        model=model,
        data=data,
        optimizer=optimizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        entity_dict=dataset.entity_dict,
        relation_dict=dataset.relation_dict,
        epochs=100 if torch.cuda.is_available() else 10,
        batch_size=config['batch_size'],
        valid_freq=10
    )

    # Evaluate the model
    metrics = train.evaluate(model, data, dataset.entity_dict, dataset.relation_dict)
    mrr = metrics['MRR']

    # Update the best configuration if this one performs better
    if mrr > best_mrr:
        best_mrr = mrr
        best_config = config

print(f"Best configuration: {best_config} with MRR: {best_mrr}")
