"""Audit sources and regimes without fitting any detector."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'lib'))
from monitoring_v3.config import load
from monitoring_v3.data import load_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    contexts, issues = load_sources(cfg, Path(cfg['output'])/'audit', strict=False)
    print(f'Audited: {len(contexts)}; source problems: {len(issues)}')
    for issue in issues:
        print(issue)
    return 1 if issues else 0


if __name__ == '__main__':
    raise SystemExit(main())
