# Community and institutional accounts

The community adds persistent posts, replies, likes, reposts, and follow relationships. Users have a public profile with activity tabs and lists of followers and followed accounts. Visitors can read public content; writing and following require a verified, approved institutional mailbox.

This implementation runs with the existing **local development service**. It has not been deployed online. The practice judge still executes Python on the host and shares a local workspace; institutional accounts do not turn it into a public, isolated, multi-user judge. Keep the server's loopback restriction until the separate deployment and judge-isolation work is complete.

## Public profiles and privacy

| Field or activity | Visibility |
| --- | --- |
| Display name, username, avatar | Always public; display name cannot be empty |
| Posts, replies, likes, reposts | Public on the author's profile |
| Followers and following | Public |
| Verified institution | Hidden by default; user controls its switch |
| Country or region | Optional and hidden by default; user controls its switch |
| Full email, provider identity, session data | Private; excluded from public API projections |
| Learning progress | Currently browser-local; not synchronized or displayed on public profiles |

Avatars currently use generated initials and a configurable color. Uploading profile photos is not implemented. The country field is a user-entered ISO country or region code; the application does not look up, infer, or publish an IP-derived location. The backend uses the same public-user projection on profiles, posts, replies, and follow lists, so turning off institution visibility also removes it from those author cards.

The learning-progress preference is saved for later progress synchronization. It does not upload existing local drafts, judge reports, or practice history. Community content and follow relationships remain public regardless of this preference.

## Configure real email verification

1. Create a Supabase project and enable the Email authentication provider, with new-user signup allowed. PaperLeet calls Supabase Auth on the server and verifies the returned user identity and confirmed email before creating a local account.
2. In Supabase's **Magic Link** email template, include `{{ .Token }}` so the recipient receives a code they can enter into PaperLeet. The default link-only template does not provide the code input flow. See [Supabase email OTP setup](https://supabase.com/docs/guides/auth/auth-email-passwordless) and [email templates](https://supabase.com/docs/guides/auth/auth-email-templates).
3. Configure custom SMTP in Supabase to send authentication messages to general users. Supabase's default email service restricts recipients to the project's team and is intended for testing. See [custom SMTP setup](https://supabase.com/docs/guides/auth/auth-smtp).
4. Review the eligible domains in [`institutions.yaml`](../institutions.yaml). Only exact listed domains can request a PaperLeet sign-in code. Verification proves control of an institutional mailbox, not current student or employee status.
5. Configure the environment in the same terminal that starts `make dev`:

```bash
cp .env.example .env
# Edit .env with your Supabase project URL and publishable key.
set -a
source .env
set +a
make dev
```

`.env` is ignored by Git. The application **does not automatically load it**; copying it alone will not enable sign-in. Restart the service after changing environment variables.

| Variable | Purpose |
| --- | --- |
| `SUPABASE_URL` | Supabase project URL, normally `https://PROJECT.supabase.co` |
| `SUPABASE_PUBLISHABLE_KEY` | Project publishable API key; the legacy `SUPABASE_ANON_KEY` is also accepted |
| `PAPERLEET_ADMIN_EMAILS` | Optional comma-separated institutional mailboxes granted local administrator role after successful verification |
| `PAPERLEET_SECURE_COOKIES` | `0` for local HTTP; `1` adds the Secure cookie attribute when an HTTPS deployment exists |

No Supabase service-role key is needed. Do not place secrets in `VITE_*` variables. The frontend talks to PaperLeet's `/api/account/` endpoints; it does not receive provider tokens. PaperLeet uses an opaque, HttpOnly session cookie with a seven-day lifetime and stores only the token's hash. Logging out revokes the current local session. Users cannot edit their verified institution or administrator role through profile settings.

If configuration is missing, the sign-in screen explains that verification is unavailable. Public community browsing and local practice remain available. There is no development login bypass or pre-verified demo account. Email sending and verification use bounded provider timeouts and per-email request limits; the HTTP adapter also limits writes.

## Review institution requests

The initial allowlist contains `mit.edu` and `berkeley.edu`, with official sources recorded alongside their entries. Other schools and research institutions can submit a name, official website, and email domain from the sign-in page. Requests are saved as intake records; submitting a request does not activate an account or approve a domain.

Inspect the request queue from the repository root:

```bash
.venv/bin/python -m paperleet.accounts institution-requests
```

This command is read-only. It prints each request and `domain_approved`, derived from the current allowlist. Use `--database /path/to/community.sqlite3` to inspect a different local database. The request's stored `status` records its intake state; domain approval is controlled by the reviewed configuration.

For an approved request, a maintainer verifies that an official institution source documents the email domain, then adds the exact domain and its evidence URL to `institutions.yaml`. Review that change as a normal repository change and restart the service. Use a stable institution ID and names in both languages where available. A domain belongs to one institution; duplicate mappings are rejected. A listed parent domain does not approve its subdomains, and `.edu` suffixes have no special privilege. Do not add a personal email provider as an institution. Unknown domains remain unable to request a sign-in code until this review is complete.

## Data and maintenance

Accounts, private email identities, session hashes, institution requests, posts, replies, reactions, and follow relationships are stored in `.paperleet/community.sqlite3`. This directory is ignored by Git. Use a SQLite-aware backup or stop the development service before copying the database and its associated journal files. The accounts and community services share the users table; no separate Supabase application tables are required for this local version.

The community supports author deletion of posts and replies, with administrator moderation enforced on the backend. Removing content also removes it from public activity results. Likes, reposts, and follows are idempotent operations; repeated requests do not inflate counts. Profile edits cannot change another user's profile or expose email through community responses.

Tests cover provider-response validation, domain gating, identity binding, session expiry and revocation, privacy projections, profile validation, and community ownership rules using mocked provider responses. **No live verification email was sent as part of implementation validation.** A configured Supabase project and an approved mailbox are still needed to verify actual delivery end to end. Public hosting, isolated multi-user judging, cloud progress synchronization, and production abuse operations remain separate work.
