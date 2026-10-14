# Run the beta from your computer with Cloudflare Tunnel

**English** | [简体中文](cloudflare-tunnel.zh-CN.md)

Use this deployment to serve `https://paperleet.com` from your computer without renting a server or opening router ports. Cloudflare provides the public HTTPS connection; Docker runs the website, accounts, database, and isolated judge on your computer. Keep the computer awake, online, and Docker running while testers use it.

```text
Browser → paperleet.com → Cloudflare → cloudflared container → web → API
                                                                   ├→ persistent app_data volume
                                                                   └→ judge manager → isolated worker
```

The local `make dev` service is for personal development. This package uses hosted mode with separate test accounts and container judging. None of its services publishes a host port. The tunnel connects to the web container, not to ports 5173 or 8765.

## 1. Prepare Docker and the domain

Start Docker Desktop on your Mac. Check that these commands work:

```sh
docker version
docker compose version
```

Use Docker Compose 2.24.4 or later. Allocate at least 4 GB of memory to Docker; the beta runs one judge job at a time, with a 2 GB worker limit. The first image build downloads Python, PyTorch, and frontend dependencies and can take several minutes.

In the Cloudflare dashboard, check that `paperleet.com` is active in your account. A domain purchased through Cloudflare already uses its nameservers. If you purchased elsewhere, follow [Cloudflare's domain setup](https://developers.cloudflare.com/dns/zone-setups/full-setup/setup/).

## 2. Create the tunnel

In your Cloudflare account, open **Networking → Tunnels → Create Tunnel**. Some dashboards show this under **Zero Trust → Networks → Tunnels**. Choose the `cloudflared` connector if prompted and name it `paperleet-beta`.

Select **Docker** in the installation instructions. Copy only the token following `--token` in the displayed command. The repository starts the connector for you, so do not also run Cloudflare's example Docker command. Keep the token private.

See the official [Tunnel setup guide](https://developers.cloudflare.com/tunnel/get-started/).

## 3. Configure and start PaperLeet

From the repository root:

```sh
cp -n deploy/tunnel.env.example deploy/.env
chmod 600 deploy/.env
openssl rand -hex 32
```

`cp -n` preserves an existing configuration. Edit `deploy/.env` locally. Set the following values, using the random output from the last command for the judge token:

```dotenv
PAPERLEET_DOMAIN=paperleet.com
PAPERLEET_AUTH_MODE=test
TUNNEL_TOKEN=YOUR_CLOUDFLARE_TUNNEL_TOKEN
PAPERLEET_JUDGE_TOKEN=YOUR_RANDOM_64_CHARACTER_HEX_TOKEN
SUPABASE_URL=
SUPABASE_PUBLISHABLE_KEY=
PAPERLEET_INVITE_EMAILS=
PAPERLEET_ADMIN_EMAILS=
```

The Cloudflare token and judge token are different credentials. `deploy/.env` is ignored by Git and excluded from image builds. Do not commit it or paste it into issues or chat.

```sh
./deploy/tunnel.sh build
./deploy/tunnel.sh test-users create tester1 --name "Tester 1"
./deploy/tunnel.sh up
./deploy/tunnel.sh status
```

Save the generated test username and access code privately. Create a separate account for each tester. Accounts previously created by the local Python CLI are in `.paperleet/`; they are not automatically copied into this Docker deployment.

The wrapper uses the `paperleet-tunnel` Compose project by default, separate from the direct server deployment. Always use `tunnel.sh` to manage this installation. If you explicitly set `COMPOSE_PROJECT_NAME`, use the same value for every operation.

## 4. Connect paperleet.com

Wait until the Cloudflare dashboard shows the connector as **Healthy**. Under the tunnel's **Routes**, add a **Published application** with:

| Setting | Value |
| --- | --- |
| Subdomain | Leave blank |
| Domain | `paperleet.com` |
| Path | Leave blank |
| Service type | `HTTP` |
| Service URL | `web:80` (full URL: `http://web:80`) |

Keep the original public Host header; leave HTTP Host Header overrides empty. `web` is the Docker service name. `localhost` inside the connector would refer to the connector itself.

Saving the route makes the running application publicly reachable. Cloudflare creates the corresponding DNS route; you do not need an A record pointing at your home IP. If a record already occupies the same hostname, resolve that specific conflict in DNS before saving. This setup serves the root domain; it does not automatically add `www`.

The browser uses HTTPS. The HTTP hop above stays inside Docker after the encrypted tunnel terminates, so Caddy does not need a public certificate or inbound ports for this setup.

## 5. Verify access and judging

Open `https://paperleet.com` from another computer or from a phone using mobile data. Sign in with a test account, save a draft, and run a problem. Check:

```sh
curl --fail https://paperleet.com/api/health
./deploy/tunnel.sh logs
```

The health response should include `"mode":"hosted"`, `"auth_required":true`, and `"judge":{"ready":true,...}`. Docker's running status alone does not prove that the public hostname and tunnel are working.

The tunnel profile limits each judge worker to 90 seconds and waits up to 110 seconds for its report, leaving room before [Cloudflare's proxy timeout](https://developers.cloudflare.com/fundamentals/reference/connection-limits/). Normal problem tests are unchanged. An unusually slow machine or an infinite loop can hit this deadline.

## Stop, resume, update, and back up

```sh
# Stop public access and the application; preserve data
./deploy/tunnel.sh down

# Resume
./deploy/tunnel.sh up

# Back up accounts, posts, drafts, and judging history
./deploy/tunnel.sh backup

# Update after backing up
git pull --ff-only
./deploy/tunnel.sh build
./deploy/tunnel.sh up
```

Application data lives in the Docker `app_data` volume, not in GitHub or Cloudflare. Backups are private files under `deploy/backups/`; copy them to another private location. Rebuilding images and `down` preserve the volume. Do not delete Docker volumes or reset Docker Desktop without first backing up. See the [hosting guide](hosting.md#backup-and-restore) for restoration, using `tunnel.sh` for this deployment.

For a temporary test session, `caffeinate -i` in a separate Mac terminal can prevent idle sleep while it runs; press Ctrl-C to stop it. Keep the lid open. Closing the lid, losing the network, quitting Docker, or turning off the computer interrupts access.

If the tunnel is unhealthy, check Docker and `tunnel.sh logs`. A 502 usually means the route cannot reach `http://web:80`; verify that URL and the running web container. For 403 responses, check that the public hostname matches `PAPERLEET_DOMAIN`. Keep API caching disabled and Cloudflare's Pseudo IPv4 header-overwrite mode off so account requests retain their normal behavior.
