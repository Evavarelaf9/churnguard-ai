# syntax=docker/dockerfile:1
# Imagem da API (também usada pelos serviços `bootstrap` e `jobs`).
# Contexto de build: raiz do repositório.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependências primeiro (camada reaproveitada enquanto requirements.txt não mudar)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Código: pacote de ML (atributos, modelo, explicabilidade, drift), API e migrations
COPY ml/__init__.py ml/__init__.py
COPY ml/src ml/src
COPY ml/models ml/models
COPY backend backend
COPY database database

# Usuário sem privilégios; pasta de dados gravável (dataset gerado pelo bootstrap)
RUN useradd --create-home --uid 10001 app \
    && mkdir -p ml/data/raw ml/data/processed ml/reports \
    && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
