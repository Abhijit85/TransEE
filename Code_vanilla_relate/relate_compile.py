import json
import math
import os
import re
from collections import defaultdict

import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
except Exception:
    sparse = None
    lsqr = None


def read_dict(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            idx, key = line.split('\t')
            mapping[key] = int(idx)
    return mapping


def read_triples(path, entity2id, relation2id):
    triples = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def build_relation_maps(triples, nrelation):
    tails_by_head = [defaultdict(set) for _ in range(nrelation)]
    heads_by_tail = [defaultdict(set) for _ in range(nrelation)]
    for h, r, t in triples:
        tails_by_head[r][h].add(t)
        heads_by_tail[r][t].add(h)
    return tails_by_head, heads_by_tail


def build_adjacency(triples):
    adjacency = defaultdict(list)
    for h, r, t in triples:
        adjacency[h].append((r, t))
    return adjacency


def mine_symmetry(triples, nrelation, threshold):
    counts = [0] * nrelation
    sym_counts = [0] * nrelation
    triple_set = set(triples)
    for h, r, t in triples:
        counts[r] += 1
        if (t, r, h) in triple_set:
            sym_counts[r] += 1
    symmetric = []
    for r in range(nrelation):
        if counts[r] == 0:
            continue
        score = sym_counts[r] / counts[r]
        if score >= threshold:
            symmetric.append({'relation': r, 'score': score})
    return symmetric


def mine_inverse(triples, nrelation, threshold):
    counts = [0] * nrelation
    pair_counts = defaultdict(int)
    triple_set = set(triples)
    for h, r, t in triples:
        counts[r] += 1
        pair_counts[(r, h, t)] += 1

    inverse_pairs = []
    for r1 in range(nrelation):
        if counts[r1] == 0:
            continue
        for r2 in range(nrelation):
            if counts[r2] == 0:
                continue
            match = 0
            for h, r, t in triples:
                if r != r1:
                    continue
                if (t, r2, h) in triple_set:
                    match += 1
            score = match / counts[r1] if counts[r1] else 0.0
            if score >= threshold:
                inverse_pairs.append({'r1': r1, 'r2': r2, 'score': score})
    return inverse_pairs


def mine_compositions(triples, nrelation, threshold, top_k=5, max_pairs_per_relpair=200000):
    tails_by_head, heads_by_tail = build_relation_maps(triples, nrelation)
    triple_set = set(triples)

    compositions = []
    for r1 in range(nrelation):
        for r2 in range(nrelation):
            # Build two-hop pairs (h, t) for r1 then r2
            pair_set = set()
            for x, heads in heads_by_tail[r1].items():
                tails = tails_by_head[r2].get(x)
                if not tails:
                    continue
                for h in heads:
                    for t in tails:
                        pair_set.add((h, t))
                        if len(pair_set) >= max_pairs_per_relpair:
                            break
                    if len(pair_set) >= max_pairs_per_relpair:
                        break
                if len(pair_set) >= max_pairs_per_relpair:
                    break
            if not pair_set:
                continue
            denom = len(pair_set)
            if denom == 0:
                continue
            # Count support for each r3
            support = defaultdict(int)
            for (h, t) in pair_set:
                for r3 in range(nrelation):
                    if (h, r3, t) in triple_set:
                        support[r3] += 1
            if not support:
                continue
            scored = []
            for r3, sup in support.items():
                conf = sup / denom
                if conf >= threshold:
                    scored.append((conf, r3, sup, denom))
            scored.sort(reverse=True)
            for conf, r3, sup, denom in scored[:top_k]:
                compositions.append({
                    'r1': r1,
                    'r2': r2,
                    'r3': r3,
                    'score': conf,
                    'support': sup,
                    'denom': denom
                })
    return compositions


def build_domain_range(triples, nrelation):
    domain = [set() for _ in range(nrelation)]
    range_ = [set() for _ in range(nrelation)]
    for h, r, t in triples:
        domain[r].add(h)
        range_[r].add(t)
    return domain, range_


def write_schema(output_path, payload):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(payload, fout, indent=2)


def solve_phase(nentity, nrelation, triples, inverse_pairs, compositions, dim,
                weight_triple=1.0, weight_inverse=1.0, weight_comp=1.0):
    if sparse is None or lsqr is None:
        raise RuntimeError('scipy is required for phase solving (pip install scipy).')

    # Build linear system A x = 0 with gauge fix (entity 0 = 0)
    # Variables: [entity phases | relation phases]
    nvars = nentity + nrelation
    rows = []
    cols = []
    data = []
    b = []

    def add_row(coeffs, rhs, weight):
        row_idx = len(b)
        for c, v in coeffs:
            rows.append(row_idx)
            cols.append(c)
            data.append(weight * v)
        b.append(weight * rhs)

    # Triple constraints: phi_t - phi_h - phi_r = 0
    for h, r, t in triples:
        add_row([(t, 1.0), (h, -1.0), (nentity + r, -1.0)], 0.0, weight_triple)

    # Inverse constraints: phi_r2 + phi_r1 = 0
    for inv in inverse_pairs:
        r1 = inv['r1']
        r2 = inv['r2']
        add_row([(nentity + r1, 1.0), (nentity + r2, 1.0)], 0.0, weight_inverse * inv['score'])

    # Composition constraints: phi_r3 - phi_r1 - phi_r2 = 0
    for comp in compositions:
        r1, r2, r3 = comp['r1'], comp['r2'], comp['r3']
        add_row([(nentity + r3, 1.0), (nentity + r1, -1.0), (nentity + r2, -1.0)],
                0.0, weight_comp * comp['score'])

    # Gauge fix: phi_entity0 = 0
    add_row([(0, 1.0)], 0.0, 1.0)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(len(b), nvars))
    b = np.array(b, dtype=np.float64)

    # Solve least squares once, then tile across dimensions
    x = lsqr(A, b, atol=1e-6, btol=1e-6, iter_lim=500)[0]
    x = x.astype(np.float32)

    # Wrap phases into (-pi, pi]
    x = (x + np.pi) % (2 * np.pi) - np.pi
    phase = np.tile(x[None, :], (dim, 1)).T  # [nvars, dim]

    entity_phase = phase[:nentity]
    relation_phase = phase[nentity:]
    return entity_phase, relation_phase


def solve_modulus(triples, nentity, nrelation, dim):
    # Deterministic modulus from degree statistics
    deg_out = np.zeros(nentity, dtype=np.float32)
    deg_in = np.zeros(nentity, dtype=np.float32)
    rel_out = np.zeros(nrelation, dtype=np.float32)
    rel_in = np.zeros(nrelation, dtype=np.float32)

    for h, r, t in triples:
        deg_out[h] += 1
        deg_in[t] += 1
        rel_out[r] += 1
        rel_in[r] += 1

    ent_strength = np.log1p(deg_out + deg_in)
    rel_strength = np.log1p(rel_out + rel_in)

    # Normalize to mean 1
    ent_strength = ent_strength / max(ent_strength.mean(), 1e-6)
    rel_strength = rel_strength / max(rel_strength.mean(), 1e-6)

    entity_mod = np.tile(ent_strength[:, None], (1, dim)).astype(np.float32)
    relation_mod = np.tile(rel_strength[:, None], (1, dim)).astype(np.float32)
    return entity_mod, relation_mod


ATOM_RE = re.compile(r'([^\(]+)\(([^,]+),([^\)]+)\)')


def _map_relation(token, relation2id):
    name = token.strip()
    if name.startswith('r_'):
        name = name[2:]
    if name in relation2id:
        return relation2id[name]
    if f'_{name}' in relation2id:
        return relation2id[f'_{name}']
    return None


def _map_entity(token, entity2id):
    name = token.strip()
    if name in entity2id:
        return entity2id[name]
    if (name.startswith('e') or name.startswith('E')) and name[1:] in entity2id:
        return entity2id[name[1:]]
    return None


def _parse_term(token, entity2id):
    token = token.strip()
    if len(token) == 1 and token.isupper():
        return ('var', token)
    ent = _map_entity(token, entity2id)
    if ent is not None:
        return ('const', ent)
    # Unknown constant; keep raw for debugging
    return ('const_raw', token)


def parse_anyburl_rules(path, entity2id, relation2id, min_conf=0.0, max_rules=None):
    rules = []
    if not path or not os.path.exists(path):
        return rules
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            try:
                conf = float(parts[2])
            except ValueError:
                continue
            if conf < min_conf:
                continue
            rule_str = parts[3]
            if '<=' not in rule_str:
                continue
            head_str, body_str = rule_str.split('<=', 1)
            head_atoms = ATOM_RE.findall(head_str)
            if not head_atoms:
                continue
            head_rel_tok, head_l, head_r = head_atoms[0]
            head_rel = _map_relation(head_rel_tok, relation2id)
            if head_rel is None:
                continue
            head_left = _parse_term(head_l, entity2id)
            head_right = _parse_term(head_r, entity2id)
            if head_left[0] == 'const_raw' or head_right[0] == 'const_raw':
                continue

            body_atoms = []
            for rel_tok, l_tok, r_tok in ATOM_RE.findall(body_str):
                rel_id = _map_relation(rel_tok, relation2id)
                if rel_id is None:
                    body_atoms = []
                    break
                left = _parse_term(l_tok, entity2id)
                right = _parse_term(r_tok, entity2id)
                if left[0] == 'const_raw' or right[0] == 'const_raw':
                    body_atoms = []
                    break
                body_atoms.append((rel_id, left, right))
            if not body_atoms:
                continue

            rules.append({
                'head_rel': head_rel,
                'head_left': head_left,
                'head_right': head_right,
                'body': body_atoms,
                'confidence': conf
            })
            if max_rules and len(rules) >= max_rules:
                break
    return rules


def _extend_bindings(atom, bindings, tails_by_head, heads_by_tail):
    rel_id, left, right = atom

    def resolve(term):
        if term[0] == 'const':
            return term[1]
        if term[0] == 'var' and term[1] in bindings:
            return bindings[term[1]]
        return None

    left_val = resolve(left)
    right_val = resolve(right)

    if left_val is not None and right_val is not None:
        if right_val in tails_by_head[rel_id].get(left_val, set()):
            return [bindings]
        return []

    if left_val is not None:
        results = []
        for cand in tails_by_head[rel_id].get(left_val, set()):
            new_bind = dict(bindings)
            if right[0] == 'var':
                new_bind[right[1]] = cand
            results.append(new_bind)
        return results

    if right_val is not None:
        results = []
        for cand in heads_by_tail[rel_id].get(right_val, set()):
            new_bind = dict(bindings)
            if left[0] == 'var':
                new_bind[left[1]] = cand
            results.append(new_bind)
        return results

    return []


def apply_rule_tail(rule, h, tails_by_head, heads_by_tail, max_bindings=2000):
    head_left = rule['head_left']
    head_right = rule['head_right']
    if head_left[0] == 'const' and head_left[1] != h:
        return []
    if head_left[0] == 'var':
        bindings = {head_left[1]: h}
    else:
        bindings = {}

    bindings_list = [bindings]
    for atom in rule['body']:
        next_list = []
        for bind in bindings_list:
            next_list.extend(_extend_bindings(atom, bind, tails_by_head, heads_by_tail))
            if len(next_list) >= max_bindings:
                break
        bindings_list = next_list
        if not bindings_list:
            break

    candidates = set()
    for bind in bindings_list:
        if head_right[0] == 'const':
            candidates.add(head_right[1])
        elif head_right[0] == 'var' and head_right[1] in bind:
            candidates.add(bind[head_right[1]])
    return list(candidates)


def build_relation_stats(triples, nrelation):
    range_sets = [set() for _ in range(nrelation)]
    tail_counts = [defaultdict(int) for _ in range(nrelation)]
    for h, r, t in triples:
        range_sets[r].add(t)
        tail_counts[r][t] += 1
    return range_sets, tail_counts


def generate_candidates_with_rules(h, r, tails_by_head, heads_by_tail, adjacency, train_triples,
                                   rules_by_rel, candidate_max=15000, topk_fallback=1000,
                                   rule_topk=None):
    range_sets, tail_counts = build_relation_stats(train_triples, len(tails_by_head))

    candidates = {}

    # Rule candidates
    for rule in rules_by_rel.get(r, []):
        cands = apply_rule_tail(rule, h, tails_by_head, heads_by_tail)
        for t in cands:
            candidates[t] = candidates.get(t, 0.0) + rule['confidence']

    # Rule-only cap before adding fallbacks
    if rule_topk is not None and len(candidates) > rule_topk:
        sorted_items = sorted(candidates.items(), key=lambda x: -x[1])[:rule_topk]
        candidates = dict(sorted_items)

    # Fallback: range + top frequent tails
    for t in range_sets[r]:
        candidates.setdefault(t, 0.0)
    top_tails = sorted(tail_counts[r].items(), key=lambda x: -x[1])[:topk_fallback]
    for t, _ in top_tails:
        candidates.setdefault(t, 0.0)

    # Fallback: 2-hop neighbors (any relation)
    two_hop = set()
    for _, mid in adjacency.get(h, []):
        for _, t2 in adjacency.get(mid, []):
            two_hop.add(t2)
    for t in list(two_hop)[:topk_fallback]:
        candidates.setdefault(t, 0.0)

    # Cap to max size by rule score
    if len(candidates) > candidate_max:
        sorted_items = sorted(candidates.items(), key=lambda x: -x[1])[:candidate_max]
        candidates = dict(sorted_items)

    return candidates


def evaluate_hybrid(triples, all_true_triples, entity_phase, relation_phase,
                    entity_mod, relation_mod, nentity, rules, train_triples,
                    alpha=0.9, candidate_max=15000, lambda_mod=1.0,
                    fallback_topk=1000, rule_topk=None):
    nrelation = max(r for _, r, _ in train_triples) + 1 if train_triples else 0
    tails_by_head, heads_by_tail = build_relation_maps(train_triples, nrelation)
    adjacency = build_adjacency(train_triples)
    rules_by_rel = defaultdict(list)
    for rule in rules:
        rules_by_rel[rule['head_rel']].append(rule)

    triple_set = set(all_true_triples)
    logs = []
    hit_in_candidates = 0
    total_candidates = 0
    for h, r, t in triples:
        rule_candidates = generate_candidates_with_rules(
            h,
            r,
            tails_by_head,
            heads_by_tail,
            adjacency,
            train_triples,
            rules_by_rel,
            candidate_max=candidate_max,
            topk_fallback=fallback_topk,
            rule_topk=rule_topk
        )
        candidates = np.array(sorted(rule_candidates.keys()), dtype=np.int64)
        total_candidates += len(candidates)
        if t in rule_candidates:
            hit_in_candidates += 1
        s_rule = np.array([rule_candidates[c] for c in candidates], dtype=np.float32)

        s_rel = score_tail_batch(entity_phase, relation_phase, entity_mod, relation_mod,
                                 h, r, candidates, lambda_mod=lambda_mod)
        if len(s_rel) > 1:
            s_rel_norm = (s_rel - s_rel.mean()) / (s_rel.std() + 1e-6)
        else:
            s_rel_norm = s_rel
        s_final = alpha * s_rule + (1.0 - alpha) * (1.0 / (1.0 + np.exp(-s_rel_norm)))

        # Filter other true tails
        for idx, cand in enumerate(candidates):
            if cand != t and (h, r, int(cand)) in triple_set:
                s_final[idx] = -1e9

        if t not in candidates:
            rank = len(candidates) + 1
        else:
            t_idx = int(np.where(candidates == t)[0][0])
            rank = 1 + int(np.sum(s_final > s_final[t_idx]))
        logs.append({
            'MRR': 1.0 / rank,
            'MR': float(rank),
            'HITS@1': 1.0 if rank <= 1 else 0.0,
            'HITS@3': 1.0 if rank <= 3 else 0.0,
            'HITS@10': 1.0 if rank <= 10 else 0.0,
        })
    metrics = {k: sum(d[k] for d in logs) / len(logs) for k in logs[0]}
    metrics['CANDIDATE_RECALL'] = hit_in_candidates / max(len(logs), 1)
    metrics['CANDIDATE_AVG_SIZE'] = total_candidates / max(len(logs), 1)
    return metrics


def score_tail_batch(entity_phase, relation_phase, entity_mod, relation_mod,
                     h, r, tails, lambda_mod=1.0, rule_bonus=None):
    # RELATE-style score (phase+modulus residual)
    phi_h = entity_phase[h]
    phi_r = relation_phase[r]
    phi_t = entity_phase[tails]
    phase_res = np.abs(np.sin((phi_h + phi_r - phi_t) / 2.0)).sum(axis=1)

    m_h = entity_mod[h]
    m_r = relation_mod[r]
    m_t = entity_mod[tails]
    mod_res = np.abs(m_h * m_r - m_t).sum(axis=1)

    score = -phase_res - lambda_mod * mod_res
    if rule_bonus is not None:
        score = score + rule_bonus
    return score


def evaluate_filtered(triples, all_true_triples, entity_phase, relation_phase,
                      entity_mod, relation_mod, nentity, lambda_mod=1.0):
    # Filtered ranking over all entities
    triple_set = set(all_true_triples)
    logs = []
    for h, r, t in triples:
        tails = np.arange(nentity, dtype=np.int64)
        scores = score_tail_batch(entity_phase, relation_phase, entity_mod, relation_mod,
                                  h, r, tails, lambda_mod=lambda_mod)
        # Filter other true tails
        for cand in range(nentity):
            if cand != t and (h, r, cand) in triple_set:
                scores[cand] = -1e9
        # Rank
        rank = 1 + int(np.sum(scores > scores[t]))
        logs.append({
            'MRR': 1.0 / rank,
            'MR': float(rank),
            'HITS@1': 1.0 if rank <= 1 else 0.0,
            'HITS@3': 1.0 if rank <= 3 else 0.0,
            'HITS@10': 1.0 if rank <= 10 else 0.0,
        })
    metrics = {k: sum(d[k] for d in logs) / len(logs) for k in logs[0]}
    return metrics


def _build_rule_index(schema, nrelation):
    symmetric_set = set(item['relation'] for item in schema.get('symmetric', []))
    inverse_map = defaultdict(list)
    for item in schema.get('inverse_pairs', []):
        inverse_map[item['r1']].append(item['r2'])
    comp_map = defaultdict(list)
    for item in schema.get('compositions', []):
        comp_map[item['r3']].append((item['r1'], item['r2']))
    domain = schema.get('domain', [[] for _ in range(nrelation)])
    range_ = schema.get('range', [[] for _ in range(nrelation)])
    return symmetric_set, inverse_map, comp_map, domain, range_


def _generate_candidates(h, r, tails_by_head, heads_by_tail, schema, nentity):
    nrelation = len(tails_by_head)
    symmetric_set, inverse_map, comp_map, domain, range_ = _build_rule_index(schema, nrelation)

    candidates = set()
    # Typed range filter
    if r < len(range_) and range_[r]:
        candidates.update(range_[r])

    # Symmetry: r(h,t) => r(t,h)
    if r in symmetric_set:
        for cand in heads_by_tail[r].get(h, set()):
            candidates.add(cand)

    # Inverse: r1(h,t) <=> r2(t,h)
    for inv_r in inverse_map.get(r, []):
        for cand in heads_by_tail[inv_r].get(h, set()):
            candidates.add(cand)

    # Composition: r1(h,x) & r2(x,t) => r3(h,t) where r3 == r
    for r1, r2 in comp_map.get(r, []):
        for x in tails_by_head[r1].get(h, set()):
            for cand in tails_by_head[r2].get(x, set()):
                candidates.add(cand)

    if not candidates:
        # Fallback to full entity set
        candidates = set(range(nentity))

    return np.array(sorted(candidates), dtype=np.int64)


def evaluate_candidates(triples, all_true_triples, entity_phase, relation_phase,
                        entity_mod, relation_mod, nentity, schema, train_triples,
                        lambda_mod=1.0):
    nrelation = len(schema.get('range', []))
    if nrelation == 0 and train_triples:
        nrelation = max(r for _, r, _ in train_triples) + 1
    tails_by_head, heads_by_tail = build_relation_maps(train_triples, nrelation)

    triple_set = set(all_true_triples)
    logs = []
    hit_in_candidates = 0
    total_candidates = 0
    for h, r, t in triples:
        candidates = _generate_candidates(h, r, tails_by_head, heads_by_tail, schema, nentity)
        total_candidates += len(candidates)
        if t in candidates:
            hit_in_candidates += 1
        scores = score_tail_batch(entity_phase, relation_phase, entity_mod, relation_mod,
                                  h, r, candidates, lambda_mod=lambda_mod)
        # Filter other true tails
        for idx, cand in enumerate(candidates):
            if cand != t and (h, r, int(cand)) in triple_set:
                scores[idx] = -1e9
        # If target not in candidate set, treat as worst rank
        if t not in candidates:
            rank = len(candidates) + 1
        else:
            t_idx = int(np.where(candidates == t)[0][0])
            rank = 1 + int(np.sum(scores > scores[t_idx]))
        logs.append({
            'MRR': 1.0 / rank,
            'MR': float(rank),
            'HITS@1': 1.0 if rank <= 1 else 0.0,
            'HITS@3': 1.0 if rank <= 3 else 0.0,
            'HITS@10': 1.0 if rank <= 10 else 0.0,
        })
    metrics = {k: sum(d[k] for d in logs) / len(logs) for k in logs[0]}
    metrics['CANDIDATE_RECALL'] = hit_in_candidates / max(len(logs), 1)
    metrics['CANDIDATE_AVG_SIZE'] = total_candidates / max(len(logs), 1)
    return metrics


def compile_schema(data_path, thresholds, output_path):
    entity2id = read_dict(os.path.join(data_path, 'entities.dict'))
    relation2id = read_dict(os.path.join(data_path, 'relations.dict'))

    train = read_triples(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    nrelation = len(relation2id)

    symmetric = mine_symmetry(train, nrelation, thresholds['symmetry'])
    inverse_pairs = mine_inverse(train, nrelation, thresholds['inverse'])
    compositions = mine_compositions(
        train,
        nrelation,
        thresholds['composition'],
        top_k=thresholds['comp_topk'],
        max_pairs_per_relpair=thresholds['comp_max_pairs']
    )

    domain, range_ = build_domain_range(train, nrelation)
    domain = [sorted(list(s)) for s in domain]
    range_ = [sorted(list(s)) for s in range_]

    payload = {
        'thresholds': thresholds,
        'symmetric': symmetric,
        'inverse_pairs': inverse_pairs,
        'compositions': compositions,
        'domain': domain,
        'range': range_
    }
    write_schema(output_path, payload)
    return payload


def compile_solver(data_path, schema, output_dir, dim):
    entity2id = read_dict(os.path.join(data_path, 'entities.dict'))
    relation2id = read_dict(os.path.join(data_path, 'relations.dict'))

    train = read_triples(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    nentity = len(entity2id)
    nrelation = len(relation2id)

    entity_phase, relation_phase = solve_phase(
        nentity,
        nrelation,
        train,
        schema['inverse_pairs'],
        schema['compositions'],
        dim=dim
    )

    entity_mod, relation_mod = solve_modulus(train, nentity, nrelation, dim)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'entity_phase.npy'), entity_phase)
    np.save(os.path.join(output_dir, 'relation_phase.npy'), relation_phase)
    np.save(os.path.join(output_dir, 'entity_modulus.npy'), entity_mod)
    np.save(os.path.join(output_dir, 'relation_modulus.npy'), relation_mod)

    return {
        'entity_phase': entity_phase,
        'relation_phase': relation_phase,
        'entity_modulus': entity_mod,
        'relation_modulus': relation_mod
    }
