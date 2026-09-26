"""WildChat filtering and user-level partitioning from Appendix C."""
from collections import Counter
import json
from pathlib import Path
import random


def filter_conversations(records, exclude_hashes=(), n=7, threshold=100):
    seen = set()
    unique = []
    counts = Counter()
    for record in records:
        counts['input'] += 1
        key = record['conversation_hash']
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    counts['after_hash_deduplication'] = len(unique)
    unique = [r for r in unique if r['conversation_hash'] not in exclude_hashes]
    counts['after_overlap_removal'] = len(unique)
    english = [r for r in unique if (r.get('language') or '').lower() == 'english']
    counts['after_language_filter'] = len(english)
    grams = []
    for record in english:
        turns = record.get('conversation') or []
        if not turns or turns[0].get('role') != 'user':
            raise ValueError('Expected a nonempty conversation beginning with a user turn')
        words = turns[0]['content'].lower().split()
        grams.append([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])
    frequencies = Counter(gram for row in grams for gram in row)
    kept = [r for r, row in zip(english, grams) if not any(frequencies[g] > threshold for g in row)]
    counts['after_ngram_filter'] = len(kept)
    return kept, dict(counts)


def split_by_user(records, ratios=(.89, .05, .06), seed=42, heldout_ids=()):
    """Keep every hashed_ip in one split, including all records of held-out users."""
    if len(ratios) != 3 or any(x < 0 for x in ratios) or abs(sum(ratios)-1) > 1e-9:
        raise ValueError('Expected nonnegative train/validation/test ratios summing to one')
    groups = {}
    heldout_users = set()
    heldout_ids = set(heldout_ids)
    for record in records:
        user = record.get('hashed_ip')
        if not user:
            raise ValueError('hashed_ip is required for leakage-free user-level partitioning')
        groups.setdefault(user, []).append(record)
        if record.get('conversation_id') in heldout_ids or record['conversation_hash'] in heldout_ids:
            heldout_users.add(user)
    users = sorted(set(groups)-heldout_users)
    random.Random(seed).shuffle(users)
    train_end = int(len(users)*ratios[0])
    val_end = train_end + int(len(users)*ratios[1])
    partitions = (users[:train_end], users[train_end:val_end], users[val_end:]+sorted(heldout_users))
    return {name: [r for user in ids for r in groups[user]] for name, ids in zip(('train','val','test'), partitions)}


def preprocess(dataset_name, output_dir, ratios=(.89,.05,.06), seed=42, exclude_dataset=None, hold_out_wildbench=False):
    from datasets import load_dataset
    if dataset_name.endswith('.jsonl'):
        records = [json.loads(line) for line in Path(dataset_name).read_text().splitlines() if line.strip()]
    else:
        records = load_dataset(dataset_name, split='train')
    excluded = set()
    if exclude_dataset:
        excluded = set(load_dataset(exclude_dataset, split='train')['conversation_hash'])
    kept, stats = filter_conversations(records, exclude_hashes=excluded)
    ids = load_dataset('allenai/WildBench', name='v2', split='test')['session_id'] if hold_out_wildbench else ()
    splits = split_by_user(kept, ratios, seed, ids)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (output_dir/f'{name}.jsonl').open('w') as f:
            for row in rows:
                f.write(json.dumps(row, default=str, ensure_ascii=False)+'\n')
        stats[name] = len(rows)
    stats.update(seed=seed, split_ratios=list(ratios), split_unit='hashed_ip', dataset=dataset_name,
                 hold_out_wildbench=hold_out_wildbench)
    (output_dir/'stats.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return splits


def main(larger_corpus=False):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', default='allenai/WildChat-4.8M' if larger_corpus else 'allenai/WildChat-1M')
    parser.add_argument('--output_dir', default='./processed_data_4p8m' if larger_corpus else './processed_data')
    parser.add_argument('--train_ratio', type=float, default=.89)
    parser.add_argument('--val_ratio', type=float, default=.05)
    parser.add_argument('--test_ratio', type=float, default=.06)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_processes', type=int, default=8, help='Accepted for compatibility; filtering preserves source order')
    parser.add_argument('--hold_out_wildbench', action='store_true', help='Optional extension: reserve entire benchmark users in test')
    parser.add_argument('--no_wildbench', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.hold_out_wildbench and args.no_wildbench:
        parser.error('Conflicting WildBench options')
    preprocess(args.data_path, args.output_dir, (args.train_ratio,args.val_ratio,args.test_ratio), args.seed,
               'allenai/WildChat-1M' if larger_corpus else None, args.hold_out_wildbench)
