"""Convert the SFT intent-annotated splits to SkyRL inputs (Appendices D.3 and F.1)."""
import argparse
import json
from pathlib import Path
import random

VALID_ENV_CLASSES = ['userlm', 'userlm_baseline', 'userlm_baseline_mixture']


def _file_suffix(env_class):
    return env_class.removeprefix('userlm')


def main(out_dir, in_dir, intent_model_name, env_class, seed=42, validation_size=10000):
    import pandas as pd
    if env_class not in VALID_ENV_CLASSES or validation_size < 1:
        raise ValueError('Invalid environment or validation subset size')
    out_dir, in_dir = Path(out_dir).expanduser(), Path(in_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _file_suffix(env_class)
    rng = random.Random(seed)
    users_by_split = {}
    for split in ('test','val','train'):
        path = in_dir/f'{split}_with_intents_{intent_model_name}.jsonl'
        examples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        users = {example['hashed_ip'] for example in examples}
        if not users or None in users or '' in users:
            raise ValueError('Each split must contain valid hashed user identifiers')
        if any(users & earlier for earlier in users_by_split.values()):
            raise ValueError('The input splits share users; regenerate the SFT splits before RL')
        users_by_split[split] = users
        rows = []
        for example in examples:
            conv = [{k: t[k] for k in ('role','content')} for t in example['conversation']]
            if not conv or conv[0]['role'] != 'user' or not example.get('intent'):
                raise ValueError('A user opening and intent are required')
            extra = dict(task_desc='userlm-wildchat', conversation_hash=example['conversation_hash'],
                         hashed_ip=example['hashed_ip'], intent_model_name=intent_model_name,
                         intent=example['intent'], full_conversation=conv)
            if env_class == 'userlm_baseline_mixture':
                extra['persona_seed'] = rng.randrange(2**31)
            rows.append(dict(data_source='allenai--WildChat-1M', env_class=env_class,
                             prompt=[conv[0]], reward_spec=dict(method='rule',ground_truth=0.),extra_info=extra))
        frame = pd.DataFrame(rows)
        frame.to_parquet(out_dir/f'{split}{suffix}.parquet',index=False)
        if split == 'val':
            frame.sample(n=min(validation_size,len(frame)),random_state=seed).to_parquet(out_dir/f'sampled_val{suffix}.parquet',index=False)
        print(f'Saved {len(frame)} {split} examples for {env_class}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--intent_model_name', default='Qwen--Qwen3-32B')
    parser.add_argument('--env_class', choices=VALID_ENV_CLASSES, default='userlm')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation_size', type=int, default=10000)
    args = parser.parse_args()
    main(args.output_dir,args.input_dir,args.intent_model_name,args.env_class,args.seed,args.validation_size)
