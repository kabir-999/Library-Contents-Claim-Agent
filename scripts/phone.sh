#!/usr/bin/env bash
# Use your PHONE camera: serves the app + backend over HTTPS on your Wi-Fi (phones only allow the camera on HTTPS).
# Usage:  ./scripts/phone.sh        (Mac and phone on the same Wi-Fi)
set -e
cd "$(dirname "$0")/.."
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$IP" ] && { echo "Could not find your LAN IP. Connect to Wi-Fi."; exit 1; }
# free ports 8000/3000 (old dev servers or an earlier run of this script) so both start cleanly
for P in 8000 3000; do lsof -ti tcp:$P | xargs kill 2>/dev/null || true; done
sleep 1
mkdir -p certs
if [ ! -f certs/cert.pem ] || ! openssl x509 -in certs/cert.pem -noout -text | grep -q "$IP"; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout certs/key.pem -out certs/cert.pem \
    -subj "/CN=lca" -addext "subjectAltName=IP:$IP,DNS:localhost" 2>/dev/null
fi
echo "Mac IP: $IP"
echo
echo "ON YOUR PHONE (same Wi-Fi):"
echo "  1. Open  https://$IP:8000/api/health  -> tap Advanced -> Proceed (accept the certificate, once)"
echo "  2. Open  https://$IP:3000             -> Start -> allow camera + microphone"
echo
( cd backend && source .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 \
    --ssl-keyfile ../certs/key.pem --ssl-certfile ../certs/cert.pem ) &
BACK=$!
trap "kill $BACK 2>/dev/null" EXIT
cd frontend && npx next dev -H 0.0.0.0 -p 3000 --experimental-https \
  --experimental-https-key ../certs/key.pem --experimental-https-cert ../certs/cert.pem
