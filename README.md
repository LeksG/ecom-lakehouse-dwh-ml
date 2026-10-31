# E-commerce Lakehouse + DWH + ML

Практична частина курсової роботи «Методи та засоби інженерії даних та знань» (варіант 1, E-commerce).

## Запуск
1. Дані вже в `data/raw/`: 8 CSV-файлів набору Olist Brazilian E-Commerce без geolocation
   (джерело: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. `pip install -r requirements.txt`
3. Підняти PostgreSQL (`docker compose up -d postgres`) або вказати свій DSN у змінній `ECOM_DWH_DSN`
   (типово `postgresql+psycopg2://postgres@/ecom_dwh?host=/tmp`), створити БД `ecom_dwh`.
   Для контейнера: `postgresql+psycopg2://postgres:postgres@localhost:5433/ecom_dwh`.
4. `python run_pipeline.py` - Bronze -> Silver -> DQ -> Gold -> звіт -> ознаки -> навчання -> інференс.
5. Перегляд експериментів: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
6. Для Airflow: `docker compose up -d`, каталог `dags/` монтується в контейнер. Веб-інтерфейси:
   Airflow - http://localhost:8080, MLflow - http://localhost:5000.

`docker-compose.override.yml` підхоплюється автоматично і є обов'язковим для запуску в Docker: він задає
порт PostgreSQL 5433, MLflow 3, закріплені версії пакетів Airflow та окремий том для `lake/`
(Delta Lake не працює на bind mount у Windows).

У репозиторії є дані, результати запусків MLflow (`mlruns/`, `mlflow.db`) і наявна частина `lake/`. Повне озеро
даних створює `python run_pipeline.py`. Каталог `.venv` не входить: його замінює `pip install -r requirements.txt`.
