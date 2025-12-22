# Stop on error
$ErrorActionPreference = "Stop"

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Install Python 3.10+ and add to PATH."
    exit 1
}

# Create venv if not exists
if (-not (Test-Path ".venv")) {
    Write-Host ">> Creating virtual environment"
    python -m venv .venv
}

# Activate venv
Write-Host ">> Activating virtual environment"
. .\.venv\Scripts\Activate.ps1

# Upgrade pip and install deps
Write-Host ">> Installing dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ---------- PIPELINE ----------
Write-Host ">> Executing part 1 - Manual Decision Tree"
python src\part1_decision_tree\DecisionTree.py

Write-Host ">> Executing dataset merge"
python src\common\merge_csv.py

Write-Host ">> Executing part 2 - ML (DT, KNN, SVM)"
python src\part2_ml\dt.py
python src\part2_ml\knn.py
python src\part2_ml\svm.py

Write-Host ">> Executing part 3 - Genetic Algorithm (BinPacking3D)"
python src\part3_ga\GeneticAlgorithm.py

Write-Host ">> Executing part 4 - ACO and CLONALG (BinPacking3D)"
python src\part4_swarm_immune\ACO.py
python src\part4_swarm_immune\CLONALG.py

Write-Host ">> Execution completed successfully."