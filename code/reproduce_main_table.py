"""Recompute the released part of Table II from saved query-level errors."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

SOLVERS = {'raw':'Raw PnP','magsac':'MAGSAC++','clnet':'CLNet (frozen)',
           'oanet':'OANet (frozen)','mhgc':'MHGC','cmcgc':'CMCGC'}
EXPECTED = {
    'raw':[68.67,40.92,99.23,139.75,4.25,68.83],
    'magsac':[95.17,28.55,69.67,120.48,19.75,43.63],
    'clnet':[95.17,36.80,71.84,121.71,3.50,56.03],
    'oanet':[95.17,35.50,71.67,122.05,4.18,52.48],
    'mhgc':[95.17,21.99,48.26,83.85,19.88,39.18],
    'cmcgc':[95.17,21.75,48.18,83.75,21.17,38.99],
}
METRICS=['coverage_pct','median_m','rmse_m','p95_m','sr5_pct','f30_pct']


def quantile(values, fraction):
    values=sorted(values)
    p=(len(values)-1)*fraction
    low=int(p)
    high=min(low+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(p-low)


def compute(rows):
    keys={(r['frame_id'],r['method']) for r in rows}
    if len(keys)!=len(rows) or len(rows)!=17280:
        raise ValueError('Expected exactly 17,280 unique query-method keys')
    frames={r['frame_id'] for r in rows}
    if len(frames)!=1728 or {f[:8] for f in frames}!={'Scene_21','Scene_22','Scene_23','Scene_24'}:
        raise ValueError('Unexpected test frames or scenes')
    groups=collections.defaultdict(list)
    for row in rows:
        groups[row['method']].append(row)
    if len(groups)!=10 or any(len(group)!=1728 for group in groups.values()):
        raise ValueError('Expected ten matchers on shared frames')
    if any({r['frame_id'] for r in group}!=frames for group in groups.values()):
        raise ValueError('Matchers must share identical frames')
    output=[]
    checks=[]
    for field,label in SOLVERS.items():
        values_by_method=[]
        for group in groups.values():
            raw=[float(r[field]) for r in group]
            # Preserve the historical invalid-output convention, not just isfinite.
            valid=[x for x in raw if math.isfinite(x) and x<9999.0]
            if not valid:
                raise ValueError(f'No valid outputs for {field}')
            n=len(raw)
            values_by_method.append([100*len(valid)/n,statistics.median(valid),
                math.sqrt(sum(x*x for x in valid)/len(valid)),quantile(valid,.95),
                100*sum(x<5 for x in valid)/n,
                100*(n-len(valid)+sum(x>30 for x in valid))/n])
        metrics=[statistics.mean(v[i] for v in values_by_method) for i in range(6)]
        output.append(dict(solver=label,**dict(zip(METRICS,metrics))))
        for key,value,expected in zip(METRICS,metrics,EXPECTED[field]):
            checks.append(dict(solver=label,metric=key,value=value,paper_value=expected,
                               passed=abs(value-expected)<=.00500001))
    return output,checks


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=root/'tables/query_errors')
    parser.add_argument('--output-dir',type=Path,default=root/'tables')
    args=parser.parse_args()
    rows=load_rows(args.input)
    output,checks=compute(rows)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'main_table_verified.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    (args.output_dir/'paper_value_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(output,indent=2))
    if not all(c['passed'] for c in checks):
        raise SystemExit('Mismatch with the paper: do not publish this table')
    print('PASS: all 36 released table values match the paper rounding.')


def load_rows(path):
    files=sorted(path.glob('part_*.csv')) if path.is_dir() else [path]
    rows=[]
    for file in files:
        with file.open(encoding='utf-8-sig',newline='') as stream:
            rows.extend(csv.DictReader(stream))
    return rows


if __name__=='__main__':
    main()
