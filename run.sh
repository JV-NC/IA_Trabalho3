#!/usr/bin/env bash
set -e

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python3 not found."
    exit 1
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "The venv module is not installed."
    echo "Execute: sudo apt install python3-venv"
    exit 1
fi

echo ">> Creating virtual environment"
python3 -m venv .venv

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
