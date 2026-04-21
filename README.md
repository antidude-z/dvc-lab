# 🎓 dvc-lab

> ML-пайплайн для прогнозирования отчисления студентов с версионированием данных через DVC

## 📋 О проекте

Проект демонстрирует воспроизводимый ML-пайплайн на датасете [UCI Student Dropout](https://archive.ics.uci.edu/ml/datasets/697) с использованием:
- **DVC** — версионирование данных и пайплайнов
- **uv** — быстрый менеджер зависимостей
- **Docker** — контейнеризация модели
- **scikit-learn** — обучение модели

## 🚀 Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone https://github.com/antidude-z/dvc-lab
cd dvc-lab

# 2. Установить зависимости (требуется uv)
uv sync

# 3. Подключить DVC-хранилище (при необходимости)
dvc remote add -d myremote <URL>

# 4. Запустить весь пайплайн
uv run dvc repro -f
```

## 🔄 Структура пайплайна (`dvc.yaml`)

```
preprocess → feature_engineering → train → build_image → deploy → healthcheck
```

| Стадия | Описание |
|--------|----------|
| `preprocess` | Загрузка данных UCI, разделение на train/test |
| `feature_engineering` | Создание признаков |
| `train` | Обучение модели, сохранение `model.pkl` |
| `build_image` | Сборка Docker-образа с моделью |
| `deploy` | Деплой через `docker/deploy.sh` |
| `healthcheck` | Проверка работоспособности модели |

## 📁 Структура проекта

```
dvc-lab/
├── src/
│   ├── preprocessing.py    # Загрузка и подготовка данных
│   ├── featurize.py        # Feature engineering
│   ├── train_model.py      # Обучение модели
│   └── health_check.py     # Проверка модели
├── docker/
│   ├── app.py              # Web-приложение для инференса
│   ├── Dockerfile          # Контейнеризация сервиса
│   └── deploy.sh           # Скрипт деплоя
├── data/                   # Артефакты (управляются DVC)
├── dvc.yaml                # Описание пайплайна
├── dvc.lock                # Фиксация версий данных
├── pyproject.toml          # Зависимости (uv)
└── README.md
```

## ⚙️ Требования

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) или pip
- [DVC](https://dvc.org/doc/install)
- Docker (для стадий `build_image`/`deploy`)

## 🛠️ Линтинг и качество кода

Проект использует `ruff` для быстрой проверки:

```bash
ruff check src/
ruff format src/
```

---

> 💡 **Совет**: Все данные и модели версионируются через DVC. Для совместной работы настройте удалённое хранилище: `dvc remote add -d storage s3://my-bucket/dvc`.
