"""Train cause and risk heads from prequential features and reviewed episode labels."""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
from monitoring_v3.config import load


from interpretation_training import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--directory',required=True)
    p.add_argument('--unit',required=True);p.add_argument('--as-of',required=True)
    p.add_argument('--attach',action='store_true',help='Attach the trained heads to this experiment model export')
    p.add_argument('--features',help='Optional prequential ensemble_features.parquet for offline investigation')
    a=p.parse_args()
    print(json.dumps(train(load(a.config),a.directory,a.unit,pd.Timestamp(a.as_of),a.attach,a.features),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
