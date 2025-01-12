import torch
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Training:
    def __init__(self, model, data, optimizer, device, entity_dict, relation_dict, epochs=50, batch_size=124, valid_freq=5):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.device = device
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.epochs = epochs
        self.batch_size = batch_size
        self.valid_freq = valid_freq

    def train(self):
        train_edge_index = self.data.train_edge_index.to(self.device)
        train_edge_type = self.data.train_edge_type.to(self.device)
        valid_edge_index = self.data.valid_edge_index.to(self.device)
        valid_edge_type = self.data.valid_edge_type.to(self.device)

        best_valid_score = 0
        test_scores = None

        lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)

        for epoch in range(self.epochs):
            self.model.train()

            entities_modulus_norm = torch.norm(self.model.entity_modulus.weight.data, dim=1, keepdim=True)
            self.model.entity_modulus.weight.data = self.model.entity_modulus.weight.data / entities_modulus_norm

            num_triples = train_edge_type.size(0)
            shuffled_indices = torch.randperm(num_triples)
            shuffled_edge_index = train_edge_index[:, shuffled_indices]
            shuffled_edge_type = train_edge_type[shuffled_indices]

            negative_edge_index = self.create_corrupted_edge_index(shuffled_edge_index, shuffled_edge_type, self.data.num_entities, negative_rate=20)

            total_loss = 0
            total_size = 0

            for batch_start in range(0, num_triples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_triples)
                batch_edge_index = shuffled_edge_index[:, batch_start:batch_end]
                batch_negative_edge_index = negative_edge_index[:, batch_start:batch_end]
                batch_edge_type = shuffled_edge_type[batch_start:batch_end]

                positive_score = self.model(batch_edge_index[0], batch_edge_type, batch_edge_index[1])
                negative_score = self.model(batch_negative_edge_index[0], batch_edge_type, batch_negative_edge_index[1])

                loss = self.model.compute_loss(positive_score, negative_score)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * (batch_end - batch_start)
                total_size += batch_end - batch_start

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / total_size:.4f}")

            if (epoch + 1) % self.valid_freq == 0:
                mrr_score, mr_score, hits_at_10, hits_at_3, hits_at_1 = self.evaluate_model(valid_edge_index, valid_edge_type)

                valid_loss = self.evaluate_loss(valid_edge_index, valid_edge_type)
                lr_scheduler.step(valid_loss)

                print(f"Validation score: MRR = {mrr_score:.4f}, MR = {mr_score:.4f}, Hits@10 = {hits_at_10:.4f}, Hits@3 = {hits_at_3:.4f}, Hits@1 = {hits_at_1:.4f}")

                if mrr_score > best_valid_score:
                    best_valid_score = mrr_score
                    test_scores = self.evaluate_model(self.data.test_edge_index.to(self.device), self.data.test_edge_type.to(self.device))

        print(f"Test scores from the best model (MMR, MR, Hits@10, Hits@3, Hits@1): {test_scores}")

    def create_corrupted_edge_index(self, edge_index, edge_type, num_entities, negative_rate=1):
        # Implement the function to create corrupted edge index
        pass

    def evaluate_loss(self, edge_index, edge_type):
        self.model.eval()
        with torch.no_grad():
            positive_score = self.model(edge_index[0].to(self.device), edge_type.to(self.device), edge_index[1].to(self.device))
            loss = positive_score.mean()
        return loss.item()

    def evaluate_model(self, edge_index, edge_type, eval_batch_size=32):
        self.model.eval()
        num_triples = edge_type.size(0)
        mrr_score = 0
        mr_score = 0
        hits_at_10 = 0
        hits_at_3 = 0
        hits_at_1 = 0
        num_predictions = 0

        with torch.no_grad():
            for batch_idx in range(math.ceil(num_triples / eval_batch_size)):
                batch_start = batch_idx * eval_batch_size
                batch_end = min((batch_idx + 1) * eval_batch_size, num_triples)
                batch_edge_index = edge_index[:, batch_start:batch_end]
                batch_edge_type = edge_type[batch_start:batch_end]
                batch_size = batch_edge_type.size(0)

                all_entities = torch.arange(self.data.num_entities, device=self.device).unsqueeze(0).repeat(batch_size, 1)
                head_repeated = batch_edge_index[0, :].reshape(-1, 1).repeat(1, self.data.num_entities)
                relation_repeated = batch_edge_type.reshape(-1, 1).repeat(1, self.data.num_entities)

                head_squeezed = head_repeated.reshape(-1)
                relation_squeezed = relation_repeated.reshape(-1)
                all_entities_squeezed = all_entities.reshape(-1)

                entity_index_replaced_tail = torch.stack((head_squeezed, all_entities_squeezed))
                predictions = self.model(entity_index_replaced_tail[0], relation_squeezed, entity_index_replaced_tail[1])
                predictions = predictions.reshape(batch_size, -1)
                gt = batch_edge_index[1, :].reshape(-1, 1)

                mrr_score += self.mrr(predictions, gt)
                mr_score += self.mr(predictions, gt)
                hits_at_10 += self.hit_at_k(predictions, gt, k=10)
                hits_at_3 += self.hit_at_k(predictions, gt, k=3)
                hits_at_1 += self.hit_at_k(predictions, gt, k=1)
                num_predictions += batch_size

        mrr_score = mrr_score / num_predictions
        mr_score = mr_score / num_predictions
        hits_at_10 = hits_at_10 / num_predictions
        hits_at_3 = hits_at_3 / num_predictions
        hits_at_1 = hits_at_1 / num_predictions
        return mrr_score, mr_score, hits_at_10, hits_at_3, hits_at_1

    def mrr(self, predictions, gt):
        indices = predictions.argsort()
        return (1.0 / (indices == gt).nonzero()[:, 1].float().add(1.0)).sum().item()

    def mr(self, predictions, gt):
        indices = predictions.argsort()
        return ((indices == gt).nonzero()[:, 1].float().add(1.0)).sum().item()

    def hit_at_k(self, predictions, gt, k=10):
        zero_tensor = torch.tensor([0], device=self.device)
        one_tensor = torch.tensor([1], device=self.device)
        _, indices = predictions.topk(k=k, largest=False)
        return torch.where(indices == gt, one_tensor, zero_tensor).sum().item()
