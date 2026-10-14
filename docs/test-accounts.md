# Test accounts for the beta

**English** | [简体中文](test-accounts.zh-CN.md)

Use administrator-created test accounts when you want people to try LeetPaper without registering an institutional email. Each tester receives a login name and a random access code. No Supabase project, email delivery, or institution approval is needed in this mode. There is no public signup or shared default password.

## Hosted beta

Follow the [hosting guide](hosting.md) for the server, domain, Docker and HTTPS. In `deploy/.env`, select:

```dotenv
PAPERLEET_AUTH_MODE=test
SUPABASE_URL=
SUPABASE_PUBLISHABLE_KEY=
PAPERLEET_INVITE_EMAILS=
PAPERLEET_ADMIN_EMAILS=
```

Keep your real `PAPERLEET_DOMAIN` and a random `PAPERLEET_JUDGE_TOKEN`. Test authentication changes how users sign in; the hosted judge still uses isolated containers and HTTPS.

Build the deployment, create a separate account for each tester, and start it:

```sh
./deploy/manage.sh build
./deploy/manage.sh test-users create tester1 --name "Tester 1"
./deploy/manage.sh test-users create tester2 --name "Tester 2"
./deploy/manage.sh up
```

Each create command prints JSON containing `username` and `access_code`. Give those two values and the website URL to that tester. The access code is displayed only when created or reset; the database stores a salted hash. Codes are not included in the repository, frontend, or account-list output.

On the website, open **Account**, enter the test login name and access code, and sign in. Each hosted account has its own private drafts and run history, as well as its own community identity. Profiles and author cards display a **Test account** label. These identities do not receive an institution-verification badge.

## Manage access

```sh
# List accounts without access codes
./deploy/manage.sh test-users list

# Issue a new code and revoke existing sessions
./deploy/manage.sh test-users reset-code tester1

# Disable one account and revoke its sessions
./deploy/manage.sh test-users disable tester2

# Optional maintainer account; ordinary accounts are members
./deploy/manage.sh test-users create maintainer --name "Maintainer" --admin
```

The login name stays fixed when someone changes their public username or display name. Test accounts, their data, and credential hashes live in the application data volume and are included in its private backups.

## Try the login locally

With the existing development environment installed, run these in the same terminal:

```sh
export PAPERLEET_AUTH_MODE=test
.venv/bin/python -m paperleet.test_accounts create tester1 --name "Tester 1"
make dev
```

Create each account only once. To start again later, run `export PAPERLEET_AUTH_MODE=test` and `make dev` in the same terminal, then sign in with the existing username and access code. Open the URL printed by Vite (default: `http://127.0.0.1:5173`). Keep the terminal running while using the site; press Ctrl-C to stop both development services. Stopping the services does not delete accounts.

The local development service still keeps practice drafts in the browser and runs code on your machine. Use the hosted deployment to test private cloud practice and isolated multi-user judging. Stop and restart an already-running development service after changing the authentication mode.

## Return to institutional sign-in

`institutional` remains the default. To switch back, set `PAPERLEET_AUTH_MODE=institutional`, configure Supabase/SMTP, the institution domains and invited email list, then run `./deploy/manage.sh up`. Test accounts cannot sign in or use their sessions in institutional mode. Institutional accounts cannot use their sessions in test mode.

Changing the mode does not delete data or convert test accounts into verified institutional accounts. Their existing public posts keep the test-account label. Use a fresh deployment data volume for a clean public launch, retaining a private backup of beta data if needed.
