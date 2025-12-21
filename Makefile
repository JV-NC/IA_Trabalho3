.PHONY: setup part1 part2 part3 part4 all clean

PYTHON=.venv/bin/python

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

data:
	$(PYTHON) src/common/merge_csv.py

part1:
	$(PYTHON) src/part1_decision_tree/DecisionTree.py

part2: data
	$(PYTHON) src/part2_ml/dt.py
	$(PYTHON) src/part2_ml/knn.py
	$(PYTHON) src/part2_ml/svm.py

part3:
	$(PYTHON) src/part3_ga/GeneticAlgorithm.py

part4:
	$(PYTHON) src/part4_swarm_immune/ACO.py
	$(PYTHON) src/part4_swarm_immune/CLONALG.py

all: part1 part2 part3 part4

clean:
	rm -rf __pycache__ */__pycache__ src/**/__pycache__ .venv