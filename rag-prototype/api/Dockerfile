FROM python:3.11-slim

# Telepítjük a poetry-t
RUN pip install poetry

# Mappa létrehozás és másolás
WORKDIR /app
COPY pyproject.toml poetry.lock* ./

RUN pip install uvicorn

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY api.py .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

