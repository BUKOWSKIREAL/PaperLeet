#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
compose() { docker compose --env-file deploy/.env -f deploy/compose.yaml "$@"; }
if [ ! -f deploy/.env ]; then
    echo 'Copy deploy/.env.example to deploy/.env and configure it first.' >&2
    exit 1
fi
case "${1:-help}" in
    build)
        compose config --quiet
        docker build -f deploy/Dockerfile.judge -t paperleet-worker:local .
        compose build
        ;;
    up) compose config --quiet; compose up -d --wait ;;
    down) compose down ;;
    status) compose ps ;;
    logs) compose logs --tail 100 ;;
    backup)
        umask 077
        mkdir -p deploy/backups
        archive="deploy/backups/paperleet-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
        if [ -e "$archive" ]; then echo 'Backup already exists; retry in a moment.' >&2; exit 1; fi
        temporary="$archive.part"
        trap 'rm -f "$temporary"' EXIT HUP INT TERM
        compose exec -T api python -m paperleet.backup backup --stdout > "$temporary"
        mv "$temporary" "$archive"
        echo "$archive"
        ;;
    restore)
        if [ "$#" -ne 2 ] || [ ! -f "$2" ]; then echo 'Usage: deploy/manage.sh restore /path/to/backup.tar.gz' >&2; exit 1; fi
        # The command rejects nonempty data volumes. Use a fresh project/volume.
        compose stop api
        compose run --rm --no-deps -T api python -m paperleet.backup restore --stdin < "$2"
        echo 'Restored. Run deploy/manage.sh up to start the API.'
        ;;
    *) echo 'Usage: deploy/manage.sh {build|up|down|status|logs|backup|restore ARCHIVE}' >&2; exit 1 ;;
esac
