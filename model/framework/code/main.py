import sys
import os
import csv
import pickle

root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root)
from mc.analyzers import Predictor

input_file = sys.argv[1]
output_file = sys.argv[2]

checkpoints_dir = os.path.join(root, "..", "..", "checkpoints")

with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)
    smiles_list = [row[0] for row in reader]


def load_file(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_predictor() -> Predictor:
    dump = load_file(os.path.join(checkpoints_dir, "model.pkl"))[(0.7, 0.1, 0.2)]
    ranker = dump["ranker"]
    scaler = dump["scaler"]
    return Predictor(ranker, scaler)


predictor = get_predictor()
mc_list = predictor.predict(smiles_list)

with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["molecular_complexity"])
    for mc in mc_list:
        writer.writerow([mc])
