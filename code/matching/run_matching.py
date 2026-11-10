"""Run the historical collection worker, not the paper's learned pose model."""
import argparse
import csv
import os
from pathlib import Path
import runpy
import sys


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',required=True)
    parser.add_argument('--pairs-csv',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    methods=(root/'data/methods.txt').read_text().splitlines()
    if args.method not in methods:
        parser.error('Method is outside the released point-matcher list')
    with args.pairs_csv.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    required={'run_name','image0','image1','out'}
    # Keep the collection worker's original pair CSV contract.
    if rows and not required<=set(rows[0]):
        parser.error(f'pairs CSV requires {sorted(required)}')
    if args.dry_run:
        print(f'Collection dry-run: method={args.method}, pairs={len(rows)}; no model imported')
        return
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OPENCV_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['IMCUI_CHECKPOINTS_DIR']=str(args.checkpoints.resolve())
    os.environ.setdefault('TORCH_HOME',str(args.checkpoints.resolve()))
    os.environ.setdefault('HF_HOME',str(args.checkpoints.resolve()/'huggingface'))
    os.environ.setdefault('IMCUI_SKIP_PYCOLMAP','1')
    sys.argv=['imcui_batch_method_worker.py','--method',args.method,'--pairs-csv',str(args.pairs_csv.resolve()),'--checkpoints',str(args.checkpoints.resolve())]
    if args.resume:
        sys.argv.append('--resume')
    runpy.run_path(str(Path(__file__).with_name('imcui_batch_method_worker.py')),run_name='__main__')


if __name__=='__main__':
    main()
