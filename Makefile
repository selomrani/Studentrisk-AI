.PHONY: up down airflow-up

up:
	docker-compose up -d

down:
	docker-compose down

airflow-up:
	docker-compose up -d airflow