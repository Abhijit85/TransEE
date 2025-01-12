

def train(model, data, optimizer, device, entity_dict, relation_dict, epochs=50, batch_size=124, valid_freq=5):
    train_edge_index = data.train_edge_index.to(device)
    train_edge_type = data.train_edge_type.to(device)
    valid_edge_index = data.valid_edge_index.to(device)
    valid_edge_type = data.valid_edge_type.to(device)

    best_valid_score = 0
    valid_scores = None
    test_scores = None

    lr_scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5) # Initialize scheduler


    for epoch in range(epochs):
        model.train()

        # Normalize entity embeddings (modulus only)
        entities_modulus_norm = torch.norm(model.entity_modulus.weight.data, dim=1, keepdim=True)
        model.entity_modulus.weight.data = model.entity_modulus.weight.data / entities_modulus_norm

        # Shuffle the training data
        num_triples = train_edge_type.size(0)
        shuffled_indices = torch.randperm(num_triples)
        shuffled_edge_index = train_edge_index[:, shuffled_indices]
        shuffled_edge_type = train_edge_type[shuffled_indices]

        # negative_edge_index = create_corrupted_edge_index(shuffled_edge_index, shuffled_edge_type, data.num_entities)
        negative_edge_index = create_corrupted_edge_index(shuffled_edge_index, shuffled_edge_type, data.num_entities, negative_rate=20)  # Generate 10 negative samples per positive sample

        total_loss = 0
        total_size = 0

        for batch_start in range(0, num_triples, batch_size):
            batch_end = min(batch_start + batch_size, num_triples)
            batch_edge_index = shuffled_edge_index[:, batch_start:batch_end]
            batch_negative_edge_index = negative_edge_index[:, batch_start:batch_end]
            batch_edge_type = shuffled_edge_type[batch_start:batch_end]

            # Compute positive and negative scores for TransEEnhanced
            positive_score = model(batch_edge_index[0], batch_edge_type, batch_edge_index[1])
            negative_score = model(batch_negative_edge_index[0], batch_edge_type, batch_negative_edge_index[1])

            # Compute loss using TransEEnhanced's loss function
            loss = model.compute_loss(positive_score, negative_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (batch_end - batch_start)
            total_size += batch_end - batch_start

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / total_size:.4f}")

        # Validation at regular intervals
        if (epoch + 1) % valid_freq == 0:
            # mrr_score, mr_score, hits_at_10 = evaluate_model(
                # Introduced the Hit@1,3
            mrr_score, mr_score, hits_at_10, hits_at_3, hits_at_1 = evaluate_model(
                model, valid_edge_index, valid_edge_type, data.num_entities, device
            )

            # introduced validation loss
            valid_loss = evaluate_loss(model, valid_edge_index, valid_edge_type, device)
            lr_scheduler.step(valid_loss)

            # print(f"Validation score: MRR = {mrr_score:.4f}, MR = {mr_score:.4f}, Hits@10 = {hits_at_10:.4f}")
            print(f"Validation score: MRR = {mrr_score:.4f}, MR = {mr_score:.4f}, Hits@10 = {hits_at_10:.4f}, Hits@3 = {hits_at_3:.4f}, Hits@1 = {hits_at_1:.4f}")
            # Track best validation score
            if mrr_score > best_valid_score:
                best_valid_score = mrr_score
                # test_mrr, test_mr, test_hits_at_10 = evaluate_model(
                test_mrr, test_mr, test_hits_at_10,test_hits_at_3,test_hits_at_1 = evaluate_model(
                    model, data.test_edge_index.to(device), data.test_edge_type.to(device), data.num_entities, device
                )
                test_scores = (test_mrr, test_mr, test_hits_at_10,test_hits_at_3,test_hits_at_1)

    print(f"Test scores from the best model (MMR, MR, Hits@10): {test_scores}")


# Helper function to evaluate loss on validation set
def evaluate_loss(model, edge_index, edge_type, device):
    model.eval()
    with torch.no_grad():
        positive_score = model(edge_index[0].to(device), edge_type.to(device), edge_index[1].to(device))
        loss = positive_score.mean()  # Or any other relevant loss calculation
    return loss.item()


def evaluate_model(model, edge_index, edge_type, num_entities, device, eval_batch_size=32):
    model.eval()
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

            all_entities = torch.arange(num_entities, device=device).unsqueeze(0).repeat(batch_size, 1)
            head_repeated = batch_edge_index[0, :].reshape(-1, 1).repeat(1, num_entities)
            relation_repeated = batch_edge_type.reshape(-1, 1).repeat(1, num_entities)

            head_squeezed = head_repeated.reshape(-1)
            relation_squeezed = relation_repeated.reshape(-1)
            all_entities_squeezed = all_entities.reshape(-1)

            entity_index_replaced_tail = torch.stack((head_squeezed, all_entities_squeezed))
            predictions = model(entity_index_replaced_tail[0], relation_squeezed, entity_index_replaced_tail[1])
            predictions = predictions.reshape(batch_size, -1)
            gt = batch_edge_index[1, :].reshape(-1, 1)

            mrr_score += mrr(predictions, gt)
            mr_score += mr(predictions, gt)
            hits_at_10 += hit_at_k(predictions, gt, device=device, k=10)
            hits_at_3 += hit_at_k(predictions, gt, device=device, k=3)
            hits_at_1 += hit_at_k(predictions, gt, device=device, k=1)
            num_predictions += batch_size

    mrr_score = mrr_score / num_predictions
    mr_score = mr_score / num_predictions
    hits_at_10 = hits_at_10 / num_predictions
    hits_at_3 = hits_at_3 / num_predictions
    hits_at_1 = hits_at_1 / num_predictions
    return mrr_score, mr_score, hits_at_10, hits_at_3, hits_at_1


# Metric Functions
def mrr(predictions, gt):
    indices = predictions.argsort()
    return (1.0 / (indices == gt).nonzero()[:, 1].float().add(1.0)).sum().item()

def mr(predictions, gt):
    indices = predictions.argsort()
    return ((indices == gt).nonzero()[:, 1].float().add(1.0)).sum().item()

def hit_at_k(predictions, gt, device, k=10):
    # Generalized Hits@k calculation
    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)
    _, indices = predictions.topk(k=k, largest=False)
    return torch.where(indices == gt, one_tensor, zero_tensor).sum().item()
