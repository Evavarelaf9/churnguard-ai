# syntax=docker/dockerfile:1
# Build do React (Node) + entrega estática e proxy da API (Nginx).
# Contexto de build: raiz do repositório.
FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ENV VITE_API_URL=/api/v1
RUN npm run build

FROM nginx:1.27-alpine
# O template é processado pelo entrypoint oficial (envsubst) na inicialização:
# a chave da API fica no servidor — nunca no JavaScript entregue ao navegador.
COPY frontend/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
ENV API_UPSTREAM=api:8000 \
    API_KEY=""
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
