import pandas as pd
import argparse
import json

from tqdm import tqdm
from pathlib import Path
from vibe_check.insight_analyzer import InsightAnalyzer

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-data', type=lambda p: Path(p).absolute(),
                    help='path to CSV file containing input data')
parser.add_argument('-o', '--output', type=lambda p: Path(p).absolute(),
                    help='output path for analysis summary')
args = parser.parse_args()

analyzer = InsightAnalyzer()
data = pd.read_csv(args.input_data)
split_data = [x for _, x in data.loc[data['role'] == 'manager'].groupby(data['dlg_id'])]


def main():
    results = []
    for dlg in tqdm(split_data):
        dlg_exc = pd.concat([dlg.head(5), dlg.tail(5)])
        results.append(analyzer.get_insight(dlg_exc))

    if args.output:
        with open(args.output) as ofile:
            json.dump(results, ofile, indent=4, ensure_ascii=False)
    print(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
