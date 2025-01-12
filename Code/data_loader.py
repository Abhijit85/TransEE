class CustomDataset:
    def __init__(self, data_path: str):
        """
        Custom Dataset class for loading and processing data without PyTorch Geometric.

        Args:
            data_path (str): Path to the dataset directory.
        """

        #data_path = '/Users/abhi/GitHUB/FederatedRAG1/DataSets/FB15k-237'
        # Paths to files
        self.entity_dict_path = os.path.join(data_path, 'entities.dict')
        self.relation_dict_path = os.path.join(data_path, 'relations.dict')
        self.train_data_path = os.path.join(data_path, 'train.txt')
        self.valid_data_path = os.path.join(data_path, 'valid.txt')
        self.test_data_path = os.path.join(data_path, 'test.txt')

        # Load dictionaries and datasets
        self.entity_dict = self._read_dict(self.entity_dict_path)
        self.relation_dict = self._read_dict(self.relation_dict_path)

        self.train_data = self._read_data(self.train_data_path)
        self.valid_data = self._read_data(self.valid_data_path)
        self.test_data = self._read_data(self.test_data_path)

        self.num_entities = len(self.entity_dict)
        self.num_relations = len(self.relation_dict)

    # def _read_dict(self, file_path):
    #     """Read a dictionary file mapping strings to integers."""
    #     with open(file_path, 'r') as f:
    #         lines = f.readlines()
    #     return {line.split('\t')[0]: int(line.split('\t')[1]) for line in lines}

    def _read_dict(self, file_path: str):
        """
        Read entity / relation dict.
        Format: dict({id: entity / relation})
        """

        element_dict = {}
        with open(file_path, 'r') as f:
            for line in f:
                id_, element = line.strip().split('\t')
                element_dict[element] = int(id_)

        return element_dict

    def _read_data(self, file_path):
        """Read triples data and map to indices."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        triples = [line.strip().split('\t') for line in lines]
        return [(self.entity_dict[h], self.relation_dict[r], self.entity_dict[t]) for h, r, t in triples]

    def get_edge_indices_and_types(self, data):
        """Convert triples into edge indices and types for PyTorch tensors."""
        heads, relations, tails = zip(*data)
        edge_index = torch.tensor([heads, tails], dtype=torch.long)  # Shape: (2, num_edges)
        edge_type = torch.tensor(relations, dtype=torch.long)  # Shape: (num_edges,)
        return edge_index, edge_type

