#!/bin/sh
# Use the same storage layout and operator commands as the hosted package,
# with no published host ports and a Cloudflare connector as the public entry.
set -eu
PAPERLEET_DEPLOYMENT=tunnel
COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME:-paperleet-tunnel}
export PAPERLEET_DEPLOYMENT COMPOSE_PROJECT_NAME
exec "$(dirname "$0")/manage.sh" "$@"
