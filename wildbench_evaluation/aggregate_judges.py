"""Majority-vote checklist items across the three paper judges, then compare models."""
import argparse
import json
from pathlib import Path
from metrics import satisfaction_statistics
from wildbench_absolute_binary_checking_then_pairwise import compute_pairwise

PAPER_JUDGES = ['gpt-5-mini', 'gemini-2.5-flash', 'claude-haiku-4-5']


def aggregate(model, input_dir, output_dir, judges=PAPER_JUDGES):
    if len(judges) != 3 or len(set(judges)) != 3:
        raise ValueError('The paper uses three distinct judges')
    tables = []
    for judge in judges:
        table = {}
        path = Path(input_dir)/f"{model.replace('/', '_')}_{judge.replace('/', '_')}_details.jsonl"
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec['session_id'] in table:
                raise ValueError(f'Duplicate session in {path.name}')
            table[rec['session_id']] = rec
        tables.append(table)
    if not all(set(t) == set(tables[0]) for t in tables):
        raise ValueError('Judge files must cover exactly the same sessions; rerun missing judgments')
    output = []
    for sid in sorted(tables[0]):
        rows = [t[sid] for t in tables]
        reference = rows[0]
        if any(r['checklist'] != reference['checklist'] or r['model_output'] != reference['model_output'] for r in rows):
            raise ValueError(f'Judges did not score the same response/checklist for session {sid}')
        n = len(reference['checklist'])
        if not n or any(r.get('judge_failed') or len(r.get('checklist_results') or []) != n for r in rows):
            raise ValueError(f'Incomplete judgment for session {sid}; rerun it before aggregation')
        votes = []
        for row in rows:
            results = row['checklist_results']
            if {r['item'] for r in results} != set(range(1,n+1)) or any(type(r['satisfied']) is not bool for r in results):
                raise ValueError(f'Invalid item indexing or verdicts for session {sid}')
            votes.append({r['item']:r['satisfied'] for r in results})
        checklist_results = [dict(item=i, satisfied=sum(v[i] for v in votes)>=2) for i in range(1,n+1)]
        output.append(dict(session_id=sid, primary_tag=reference.get('primary_tag'), checklist=reference['checklist'],
                           model_input=reference.get('model_input'), model_output=reference['model_output'], checklist_results=checklist_results, judge_failed=False,
                           satisfaction_rate=sum(r['satisfied'] for r in checklist_results)/n))
    output_dir = Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    prefix = model.replace('/','_')+'_majority'
    (output_dir/f'{prefix}_details.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in output))
    summary = dict(model=model, judges=judges, aggregation='item-level majority',
                   satisfaction=satisfaction_statistics([r['satisfaction_rate'] for r in output]))
    (output_dir/f'{prefix}_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--judges', nargs=3, default=PAPER_JUDGES)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    for model in args.models:
        print(json.dumps(aggregate(model,args.input_dir,args.output_dir,args.judges),indent=2))
    for i, model1 in enumerate(args.models):
        for model2 in args.models[i+1:]:
            compute_pairwise(model1,model2,'majority',args.output_dir,args.output_dir)
