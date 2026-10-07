.PHONY: install lint format typecheck test test-integration evals up down run

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy src

test:
	pytest -m "not integration"

test-integration:
	pytest -m integration

evals:
	python -m evals.run_evals

up:
	docker compose up --build -d

down:
	docker compose down

run:
	uvicorn production_rag.api.main:app --reload
