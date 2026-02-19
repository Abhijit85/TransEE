def categorize_relations(triples):
    """
    Categorize relations into 1:1, 1:M, M:1, and M:M based on head-to-tail and tail-to-head mappings.
    """
    relation_mapping = {'1:1': set(), '1:M': set(), 'M:1': set(), 'M:M': set()}
    head_to_tail = {}
    tail_to_head = {}

    for head, relation, tail in triples:
        head_to_tail.setdefault(relation, {}).setdefault(head, set()).add(tail)
        tail_to_head.setdefault(relation, {}).setdefault(tail, set()).add(head)

    for relation in head_to_tail:
        head_counts = [len(tails) for tails in head_to_tail[relation].values()]
        tail_counts = [len(heads) for heads in tail_to_head[relation].values()]

        max_head_count = max(head_counts, default=0)
        max_tail_count = max(tail_counts, default=0)

        if max_head_count == 1 and max_tail_count == 1:
            relation_mapping['1:1'].add(relation)
        elif max_head_count > 1 and max_tail_count == 1:
            relation_mapping['M:1'].add(relation)
        elif max_head_count == 1 and max_tail_count > 1:
            relation_mapping['1:M'].add(relation)
        else:
            relation_mapping['M:M'].add(relation)

    return relation_mapping


def filter_test_triples(test_triples, relation_set):
    """
    Filter test triples to include only those with relations in the specified set.
    """
    return [triple for triple in test_triples if triple[1] in relation_set]
