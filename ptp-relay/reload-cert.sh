#!/bin/sh
# Loads a renewed certificate. The provider's download, certs/certificate.pfx
# (password in PFX_PASSWORD in .env), is unpacked into certs/certificate.pem
# - the site's certificate, the intermediate, the private key, in that order,
# as Caddy reads it - and the HTTPS front is restarted when the file changed
# since it was last loaded. Run from cron, e.g. hourly:
#   17 * * * * /home/pi/ptp-relay/reload-cert.sh
cd "$(dirname "$0")" || exit 1
pfx=certs/certificate.pfx
pem=certs/certificate.pem
loaded=certs/.loaded
umask 077

unpack() {  # $1: extra openssl option (-legacy for older exports)
    openssl pkcs12 $1 -in "$pfx" -passin env:PFX_PASSWORD -clcerts -nokeys >"$pem.new" 2>/dev/null &&
        openssl pkcs12 $1 -in "$pfx" -passin env:PFX_PASSWORD -cacerts -nokeys >>"$pem.new" 2>/dev/null &&
        openssl pkcs12 $1 -in "$pfx" -passin env:PFX_PASSWORD -nocerts -nodes >>"$pem.new" 2>/dev/null &&
        grep -q "PRIVATE KEY" "$pem.new"
}

if [ -f "$pfx" ] && { [ ! -f "$pem" ] || [ "$pfx" -nt "$pem" ]; }; then
    PFX_PASSWORD=$(sed -n 's/^PFX_PASSWORD=//p' .env)
    export PFX_PASSWORD
    if unpack "" || unpack -legacy; then
        mv "$pem.new" "$pem"
    else
        rm -f "$pem.new"
        logger -t ptp-relay "certificate.pfx does not open - is PFX_PASSWORD in .env right?"
        echo "certificate.pfx does not open - is PFX_PASSWORD in .env right?" >&2
        exit 1
    fi
fi

[ -f "$pem" ] || exit 0
if [ ! -f "$loaded" ] || [ "$pem" -nt "$loaded" ]; then
    if docker compose ps --services --status running 2>/dev/null | grep -qx https; then
        docker compose restart https >/dev/null 2>&1
    fi
    touch "$loaded"
    logger -t ptp-relay "certificate loaded: $(openssl x509 -noout -enddate -in "$pem" 2>/dev/null)"
fi
