# Text2SQL Schema Linking Service

## Описание

В больших базах данных с множеством таблиц LLM часто испытывают трудности с выбором релевантных таблиц для генерации SQL-запросов. Этот сервис использует Bi-Encoder модель для ранжирования таблиц по релевантности к вопросу пользователя, что позволяет фильтровать кандидатов перед генерацией SQL.

**Метрики:** Оптимизируем Recall@K и MRR.

## Установка и использование

### Setup

```bash
git clone <repo>
cd text2sql-schema-linking
uv sync
pre-commit install
```

### Data Management

Сначала попробуйте скачать готовые данные (хранилище Google Disk):

```bash
dvc pull
```

Если данных нет, сгенерируйте их:

```bash
python schema_linker/commands.py download
python schema_linker/commands.py preprocess
```

### Train

```bash
python schema_linker/commands.py train
```

Или с переопределением параметров:

```bash
python schema_linker/commands.py train --config_name="config"
```

### Inference

Примеры команд для инференса и экспорта в ONNX:

```bash
python schema_linker/commands.py infer --checkpoint_path="models/best-checkpoint.ckpt" --question="How many students?" --candidates="['student', 'course', 'teacher']"
python schema_linker/commands.py export --checkpoint_path="models/best-checkpoint.ckpt" --output_path="model.onnx"
```

### Docker Quickstart

Запуск: `docker compose up --build`

Эта команда поднимет MLflow сервер (доступен по http://localhost:8080) и подготовит контейнер для обучения.

Пример запуска обучения внутри контейнера:

```bash
docker compose run --rm train python schema_linker/commands.py train
```

### CI/CD

Проект использует GitHub Actions для непрерывной интеграции. CI запускается на push в main и pull requests, проверяя линтеры и базовые тесты.

Запуск тестов локально:

```bash
PYTHONPATH=. uv run pytest
```

### Тестирование

Проект включает базовые unit-тесты для проверки инициализации модели. Тесты запускаются автоматически в CI.