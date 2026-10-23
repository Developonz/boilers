"""Run audit, monthly continuation, alarm comparisons and episode catalog generation."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent/'lib'))
from monitoring_v3.config import load
from monitoring_v3.data import load_sources
from monitoring_v3.pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('incremental_settings.json')))
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    cfg = load(args.config)
    if args.audit_only:
        _, issues = load_sources(cfg, Path(cfg['output'])/'audit', strict=False)
        print(f'Source problems: {len(issues)}')
        for issue in issues:
            print(issue)
        return 1 if issues else 0
    result = run(cfg)
    print(result['status'].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
