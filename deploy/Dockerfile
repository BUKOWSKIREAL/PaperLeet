FROM node:22-alpine AS frontend
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS api
WORKDIR /app
RUN pip install --no-cache-dir PyYAML==6.0.2 && \
    groupadd --gid 10001 paperleet && useradd --uid 10001 --gid 10001 --no-create-home paperleet && \
    mkdir -p /var/lib/paperleet && chown 10001:10001 /var/lib/paperleet
ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PAPERLEET_ROOT=/app
COPY paperleet/ ./paperleet/
COPY problems/ ./problems/
COPY institutions.yaml papers.yaml ./
USER 10001:10001
CMD ["python", "-m", "paperleet.cli", "serve"]

FROM docker:28-cli AS dockercli
FROM api AS judge-manager
USER root
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
ENV HOME=/tmp
CMD ["python", "-m", "paperleet.hosting.judge_service"]

FROM caddy:2-alpine AS web
COPY --from=frontend /build/web/dist/ /srv/
COPY deploy/Caddyfile /etc/caddy/Caddyfile
