# Public scoreboard relay (ptp-relay)

Puts the live ranking on the internet from a Raspberry Pi (or any Linux box
with Docker) in the judge's network. The Touchdown Analyzer on the judge PC
**pushes** the board to the Pi; nginx in a container there keeps it and
serves it, read-only, and nothing else:

```
judge PC: Touchdown Analyzer ──PUT every 3 s, with a token──► Raspberry Pi: ptp-relay
          (any address, may stay on 127.0.0.1)                  push port :5054 (local network only)
                                                                      │  kept in memory
internet ─► gateway / public URL ─────────────────────────────► public port :5051
                                                                  /, /api/public/board, /api/public/ranking.pdf
```

- The Pi never calls the judge PC. Its address may change from day to day;
  only the Pi's must stay the same. The analyzer needs no network access
  from outside and no firewall rule.
- What is pushed is what the public board shows: the page, the board of the
  newest session (every 3 s, changed or not), each session's board and its
  ranking PDF (when they change). Slim: time, offset, points, status, pilot /
  aircraft per landing. No wheel tracks, notes, history, OGN data or file
  paths. The judge's pages - recording, review, editing, scoring rules,
  deleting sessions - never leave the judge PC.
- Viewers never reach the judge PC, however many there are.
- When the analyzer stops, the newest board is no longer renewed; after
  30 s the page says "Precision Touchdown Analyzer not active" until the
  analyzer pushes again. Days picked with `?session=` stay as they were.
- The pushed files live in memory on the Pi (no SD-card wear). After a
  restart of the relay the page says "not active" until the analyzer's next
  full push, within a minute.
- Pilot names become public. Make sure that is fine with them.

## 1. The Raspberry Pi

Raspberry Pi OS or Ubuntu, 64-bit recommended. Give it a fixed address in
the router (DHCP reservation), e.g. `192.168.0.118`.

```bash
# Docker with the compose plugin (skip if installed)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"      # log out and in again

# this folder, from the repository ...
git clone --depth 1 --filter=blob:none --sparse   git@github.com:Intrinsic-Engineering-GmbH/precision-touchdown-analyzer.git
cd precision-touchdown-analyzer && git sparse-checkout set ptp-relay && cd ptp-relay
# ... or copied over: scp -r ptp-relay pi@<pi>:~/ && cd ~/ptp-relay

cp .env.example .env && chmod 600 .env
sed -i "s/^PUSH_TOKEN=.*/PUSH_TOKEN=$(openssl rand -hex 24)/" .env
grep PUSH_TOKEN .env                 # this goes into the analyzer, step 2
docker compose up -d
```

It starts again by itself after a reboot (`restart: unless-stopped`), as its
own Compose project `ptp-relay` - other containers on the Pi are untouched.

The push port (`PUSH_PORT`, 5054) is for the local network only: the router
must not forward it. Without the token it answers 401 to everyone.

Check it on the Pi:

```bash
curl -s localhost:5051/healthz                       # ok
curl -s -o /dev/null -w "%{http_code}
" localhost:5051/            # 503 until the first push, then 200
curl -s -o /dev/null -w "%{http_code}
" -X PUT localhost:5054/board.json   # 401: no token
docker compose logs -f
```

## 2. The judge PC

In the Touchdown Analyzer's launcher, under **Public scoreboard (relay)**:

- **push to**: the Pi's push port, e.g. `http://192.168.0.118:5054`
- **token**: `PUSH_TOKEN` from the Pi's `.env`

then **Start**. They are kept in the data directory's `.env` (`RELAY_URL`,
`RELAY_TOKEN`; the environment overrides them), so `touchdown-analyzer
serve` in a terminal pushes as well. The **Public board** line under
Services shows the last push, or why it fails (Pi not reachable, token
refused).

"Reachable on the network" stays off unless phones on the field need the
judge's pages: the relay does not need it.

Then open `http://<pi>:5051/` in a browser: the scoreboard.

## 3. The gateway

Point the public URL at the Pi, port 5051 (`RELAY_PORT` in `.env`) for
plain `http://`. Never the push port.

### HTTPS

The `https` service is a small Caddy in front of the relay that serves the
board over HTTPS on the one port the router forwards - no port 80 or 443.
The certificate is the one the DNS / hosting provider issues and renews for
`PUBLIC_HOST` (e.g. a "Let's Encrypt Standard" certificate in its control
panel), downloaded as `certificate.pfx` - certificate, intermediate and
private key in one password-protected file. `reload-cert.sh` unpacks it into
`certs/certificate.pem` for Caddy (a PEM download works too: copy it there
directly).

1. Copy the file to the Pi - from the PC:

   ```
   scp certificate.pfx pi@<pi>:ptp-relay/certs/certificate.pfx
   ```

2. Put its password into `.env` as `PFX_PASSWORD=...` (`chmod 600 .env`),
   then `./reload-cert.sh` - it says so if the password does not fit. Check
   (the key is not shown): `openssl x509 -noout -subject -enddate -in certs/certificate.pem`.
3. In `.env`: `COMPOSE_PROFILES=https`, `HTTPS_PORT=5051` (the forwarded
   port) and `RELAY_PORT=5052` (plain http, local network only), then
   `docker compose up -d`. The board is at `https://<PUBLIC_HOST>:5051/`.
4. Renewals: the provider renews the certificate by itself, but the Pi does
   not learn of it - download the renewed `certificate.pfx` and copy it over
   the old one (every two months or so; the provider's list shows the date).
   The cron job below unpacks and loads it within the hour:

   ```
   chmod +x reload-cert.sh
   (crontab -l 2>/dev/null; echo "17 * * * * $HOME/ptp-relay/reload-cert.sh") | crontab -
   ```

   Should the file run out anyway, browsers warn and the board stays
   reachable over plain http on the local network (`RELAY_PORT`).

### Other setups

- **A gateway or tunnel that does TLS itself**: let it forward plain HTTP to
  the relay's port and leave the `https` service off.
- **A reverse proxy already on 80/443** (e.g. Caddy for other services):
  give the board its own public host name there and pass it to the relay -
  in a Caddyfile:

  ```
  board.example.org {
      reverse_proxy host.docker.internal:5051   # or <pi LAN address>:5051
  }
  ```

  (A Caddy container needs `extra_hosts: ["host.docker.internal:host-gateway"]`
  to reach a port on the Pi itself.)
- If the gateway or proxy sets `X-Forwarded-For`, the relay's per-viewer rate
  limit uses the real viewer address; otherwise every viewer counts as the
  gateway, and the limit (5 requests/s, bursts of 40) is shared - raise it in
  `nginx/templates/default.conf.template` if the board then shows
  "connection lost" under load.

## Addresses

| URL | what |
|---|---|
| `/` | the scoreboard (follows the newest session) |
| `/?session=2026-09-13` | a given day |
| `/?refresh=10&page=15` | refresh every 10 s, turn pages every 15 s |
| `/api/public/ranking.pdf` | the ranking as a PDF |
| `/healthz` | the relay is up |

## Updating

```bash
cd ~/ptp-relay && docker compose pull && docker compose up -d --force-recreate
```

(After a change to the relay's files, copy them over again first, or
`git pull` in the sparse checkout.)

### From the earlier version (the relay asked the judge PC)

In `.env`, delete `JUDGE_URL` and add `PUSH_TOKEN` and `PUSH_PORT` as in
step 1, then `docker compose up -d`; `docker volume rm ptp-relay_board-cache`
removes the old cache. On the judge PC, the firewall rule for port 8080 is
no longer needed:

```powershell
Remove-NetFirewallRule -DisplayName "Touchdown Analyzer - scoreboard relay"
```
