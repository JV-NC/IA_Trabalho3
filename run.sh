#!/usr/bin/env bash
set -e

echo ">> Creating virtual environment"
python -m venv .venv

echo ">> Activating environment"
source .venv/bin/activate

echo ">> Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo ">> Executing part 1 - Manual Decision Tree"
make part1

echo ">> Executing part 2 - ML (DT, KNN, SVM)"
make part2

echo ">> Executing part 3 - Genetic Algorithm (BinPacking3D)"
make part3

echo ">> Executing part 4 - ACO and CLONALG (BinPacking3D)"
make part4

echo ">> Execution completed successfully."
