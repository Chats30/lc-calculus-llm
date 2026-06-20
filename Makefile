.PHONY: install data headroom train eval figures test all clean

install:
	pip install -e .

data:
	python scripts/make_dataset.py --n 3000 --out ./data --seed 0

headroom:
	python scripts/run_headroom.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --data ./data/test_id.jsonl

train:
	mlx_lm.lora -c configs/mlx_lora.yaml


eval:
	python scripts/run_eval.py --config configs/eval.yaml

figures:
	python scripts/make_figures.py

test:
	pytest -q

all: data train eval figures
