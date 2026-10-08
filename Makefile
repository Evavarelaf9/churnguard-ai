# Atalhos do ChurnGuard AI (Linux/macOS/WSL). No Windows sem make, use os comandos equivalentes do README.
.PHONY: up down logs jobs reset data train report test test-ml test-api test-web api web

up:            ## Sobe tudo (banco, bootstrap, API, painel)
	docker compose up --build -d
down:          ## Para os contêineres (mantém os dados)
	docker compose down
reset:         ## Para e apaga os volumes (banco e dados gerados)
	docker compose down -v
logs:
	docker compose logs -f api bootstrap
jobs:          ## Rotina diária: resultado real → pontuação → monitoramento
	docker compose --profile jobs run --rm jobs

data:          ## Gera dataset e base analítica (local)
	python -m ml.src.data.generate_synthetic
	python -m ml.src.data.build_analytical_base
train:         ## Treina, avalia e salva o modelo (~8 min)
	python -m ml.src.training.train
report:        ## EDA + relatório do modelo
	python -m ml.src.eda.run_eda
	python -m ml.src.evaluation.report

api:           ## API local com recarga automática
	python -m uvicorn backend.app.main:app --reload
web:           ## Frontend local
	cd frontend && npm run dev

test: test-ml test-api test-web
test-ml:
	python -m pytest ml/tests -q
test-api:
	python -m pytest backend/tests -q
test-web:
	cd frontend && npm run test
