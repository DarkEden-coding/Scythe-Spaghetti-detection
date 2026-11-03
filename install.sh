#!/usr/bin/env bash
# Scythe installer.
#
#   curl -fsSL https://raw.githubusercontent.com/DarkEden-coding/Scythe-Spaghetti-detection/master/install.sh | bash
#
# Installs the newest release as a systemd service, then prints the address
# of the web setup. Run it as the user that runs Klipper (not root); it asks
# for sudo when it needs it. Running it again updates the install.
#
# Options:
#   --ref REF        install a branch or tag instead of the newest release
#   --dir DIR        install directory (default ~/scythe)
#   --port PORT      dashboard port (default: first free port from 8090)
#   --uninstall      remove the service and system changes, keep the files
#   --purge          with --uninstall, also delete the install directory
set -euo pipefail

REPO="https://github.com/DarkEden-coding/Scythe-Spaghetti-detection.git"
PRIMARY_BRANCH="master"
SERVICE="scythe"
HELPER_DIR="/usr/local/lib/scythe"
SUDOERS="/etc/sudoers.d/scythe"
MOONRAKER_CONF="$HOME/printer_data/config/moonraker.conf"
MOONRAKER_ASVC="$HOME/printer_data/moonraker.asvc"

REF=""
DIR=""
PORT=""
UNINSTALL=0
PURGE=0

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1;33m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        --ref) REF="${2:?--ref needs a value}"; shift 2 ;;
        --dir) DIR="${2:?--dir needs a value}"; shift 2 ;;
        --port) PORT="${2:?--port needs a value}"; shift 2 ;;
        --uninstall) UNINSTALL=1; shift ;;
        --purge) PURGE=1; shift ;;
        -h|--help) echo "Options: --ref REF, --dir DIR, --port PORT, --uninstall [--purge]"; exit 0 ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done

[[ "$(uname -s)" == "Linux" ]] || die "Scythe installs on Linux (a Klipper Pi or a Linux PC)."
[[ $EUID -ne 0 ]] || die "Run this as your normal user, not root. It asks for sudo when needed."
command -v systemctl >/dev/null || die "systemd is required."

# Reuse an existing install directory, including the old default name.
if [[ -z "$DIR" ]]; then
    DIR="$HOME/scythe"
    if [[ ! -d "$DIR/.git" && -d "$HOME/Scythe-Spaghetti-detection/.git" ]]; then
        DIR="$HOME/Scythe-Spaghetti-detection"
    fi
fi
DIR="$(realpath -m "$DIR")"

sudo_run() {
    if ! sudo -n true 2>/dev/null; then
        bold "Scythe needs sudo to install its service."
    fi
    # sudo asks for the password on the terminal itself. Keep stdin away from
    # the commands: under `curl | bash` it is the rest of this script.
    sudo "$@" </dev/null
}

# --------------------------------------------------------------------------- #
# Uninstall
# --------------------------------------------------------------------------- #

remove_moonraker_entry() {
    [[ -f "$MOONRAKER_CONF" ]] || return 0
    if grep -q "^\[update_manager $SERVICE\]" "$MOONRAKER_CONF"; then
        # Delete the section: from its header to the line before the next one.
        awk -v header="[update_manager $SERVICE]" '
            $0 == header { skip = 1; next }
            skip && /^\[/ { skip = 0 }
            !skip { print }' "$MOONRAKER_CONF" > "$MOONRAKER_CONF.scythe-tmp"
        mv "$MOONRAKER_CONF.scythe-tmp" "$MOONRAKER_CONF"
    fi
    if [[ -f "$MOONRAKER_ASVC" ]]; then
        sed -i "/^$SERVICE\$/d" "$MOONRAKER_ASVC"
    fi
    sudo_run systemctl restart moonraker 2>/dev/null || true
}

uninstall() {
    step "Removing Scythe"
    if [[ -x "$HELPER_DIR/scythe-helper" ]]; then
        sudo_run "$HELPER_DIR/scythe-helper" mainsail-unlink 2>/dev/null || true
    fi
    navi="$HOME/printer_data/config/.theme/navi.json"
    if [[ -f "$navi" ]] && grep -q '"/scythe/"' "$navi"; then
        python3 - "$navi" <<'EOF'
import json, sys, pathlib
path = pathlib.Path(sys.argv[1])
entries = [e for e in json.loads(path.read_text()) if e.get("href") != "/scythe/"]
path.write_text(json.dumps(entries, indent=2) + "\n") if entries else path.unlink()
EOF
    fi
    sudo_run systemctl disable --now "$SERVICE" 2>/dev/null || true
    sudo_run rm -f "/etc/systemd/system/$SERVICE.service" "$SUDOERS"
    sudo_run rm -rf "$HELPER_DIR"
    sudo_run systemctl daemon-reload
    remove_moonraker_entry
    if ((PURGE)); then
        rm -rf "$DIR"
        echo "Deleted $DIR."
    else
        echo "Kept $DIR (settings and model). Use --uninstall --purge to delete it too."
    fi
    echo "Tailscale was left installed; run 'sudo tailscale serve reset' to drop Scythe's HTTPS address."
    bold "Scythe is uninstalled."
}

if ((UNINSTALL)); then
    uninstall
    exit 0
fi

# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #

step "Checking prerequisites"
missing=()
for tool in git curl; do
    command -v "$tool" >/dev/null || missing+=("$tool")
done
if ((${#missing[@]})); then
    command -v apt-get >/dev/null || die "Please install: ${missing[*]}"
    sudo_run apt-get update -qq
    sudo_run apt-get install -y -qq "${missing[@]}"
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    echo "Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
command -v uv >/dev/null || die "uv did not install. See https://docs.astral.sh/uv/."

case "$(uname -m)" in
    aarch64|x86_64|amd64) LOCAL_MODEL=1 ;;
    *) LOCAL_MODEL=0
       warn "This is a 32-bit system, so the local model can't run here. Scythe will use an AI decision model (OpenRouter) instead." ;;
esac

step "Downloading Scythe"
if [[ -z "$REF" ]]; then
    REF="$(git ls-remote --tags --refs "$REPO" 'v*' | sed 's#.*refs/tags/##' \
        | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1 || true)"
    if [[ -z "$REF" ]]; then
        warn "No release has been published yet; installing the $PRIMARY_BRANCH branch."
        REF="$PRIMARY_BRANCH"
    fi
fi

if [[ -d "$DIR/.git" ]]; then
    if [[ -n "$(git -C "$DIR" status --porcelain --untracked-files=no)" ]]; then
        die "$DIR has local changes. Commit or discard them, then run the installer again."
    fi
    git -C "$DIR" fetch --quiet --tags --force origin
else
    [[ ! -e "$DIR" ]] || die "$DIR exists but isn't a Scythe checkout. Choose another with --dir."
    git clone --quiet "$REPO" "$DIR"
fi

cd "$DIR"
if git rev-parse --quiet --verify "refs/tags/$REF" >/dev/null; then
    # Keep the primary branch checked out at the release tag; Moonraker's
    # update manager expects a branch, and moves it to newer tags.
    git checkout --quiet -B "$PRIMARY_BRANCH" "refs/tags/$REF"
    git branch --quiet --set-upstream-to "origin/$PRIMARY_BRANCH" "$PRIMARY_BRANCH"
    echo "Installing release $REF."
elif git rev-parse --quiet --verify "refs/remotes/origin/$REF" >/dev/null; then
    git checkout --quiet -B "$REF" "origin/$REF"
    git branch --quiet --set-upstream-to "origin/$REF" "$REF"
    echo "Installing branch $REF ($(git rev-parse --short HEAD))."
else
    die "No branch or tag called $REF."
fi

step "Installing Python and dependencies"
if [[ ! -x .venv/bin/python ]]; then
    uv venv --quiet --seed --python 3.11 .venv
fi
# Moonraker's update manager installs requirements with the venv's own pip.
uv pip install --quiet --python .venv/bin/python --index-strategy unsafe-best-match \
    pip -r requirements.txt

if ((LOCAL_MODEL)); then
    step "Downloading the detection model (175 MB)"
    .venv/bin/python -m src download-model
fi

# Pick the dashboard port: keep a configured one, else the first free port.
current_port="$(.venv/bin/python - <<'EOF'
from src import paths
from src.config import load_settings
print(load_settings(validate=False).web.port if paths.config_path().exists() else "")
EOF
)"
if [[ -z "$PORT" ]]; then
    PORT="$current_port"
fi
if [[ -z "$PORT" ]]; then
    PORT=8090
    while ss -Hltn "sport = :$PORT" 2>/dev/null | grep -q .; do
        PORT=$((PORT + 1))
    done
fi

ON_KLIPPER=0
[[ -f "$MOONRAKER_CONF" ]] && ON_KLIPPER=1

.venv/bin/python - "$PORT" "$ON_KLIPPER" "$LOCAL_MODEL" <<'EOF'
import sys
from src import paths
from src.config import save_overrides
port, on_klipper, local_model = int(sys.argv[1]), sys.argv[2] == "1", sys.argv[3] == "1"
changes = {"web": {"port": port}}
if not paths.config_path().exists():
    # Moonraker updates Klipper hosts; elsewhere Scythe follows releases itself.
    changes["enable_auto_update"] = not on_klipper
    if not local_model:
        changes["detection"] = {"backend": "decision"}
save_overrides(changes)
EOF

step "Installing the service"
tmp_unit="$(mktemp)"
sed -e "s#@USER@#$USER#g" -e "s#@DIR@#$DIR#g" deploy/scythe.service > "$tmp_unit"
sudo_run install -m 644 "$tmp_unit" "/etc/systemd/system/$SERVICE.service"
rm -f "$tmp_unit"

# The web setup's Tailscale and Mainsail buttons run this root-owned helper,
# and only this helper, through sudo.
sudo_run install -d -m 755 "$HELPER_DIR"
sudo_run install -o root -g root -m 755 deploy/scythe-helper "$HELPER_DIR/scythe-helper"
tmp_sudoers="$(mktemp)"
echo "$USER ALL=(root) NOPASSWD: $HELPER_DIR/scythe-helper" > "$tmp_sudoers"
if sudo_run visudo -cqf "$tmp_sudoers"; then
    sudo_run install -o root -g root -m 440 "$tmp_sudoers" "$SUDOERS"
else
    warn "Could not add the sudoers rule; the setup's Tailscale and Mainsail buttons will be off."
fi
rm -f "$tmp_sudoers"

sudo_run systemctl daemon-reload
sudo_run systemctl enable --quiet "$SERVICE"
sudo_run systemctl restart "$SERVICE"

if ((ON_KLIPPER)); then
    step "Registering with Moonraker's update manager"
    if ! grep -q "^\[update_manager $SERVICE\]" "$MOONRAKER_CONF"; then
        cat >> "$MOONRAKER_CONF" <<EOF

[update_manager $SERVICE]
type: git_repo
channel: stable
path: $DIR
origin: $REPO
primary_branch: $PRIMARY_BRANCH
virtualenv: $DIR/.venv
requirements: requirements.txt
managed_services: $SERVICE
info_tags:
  desc=Scythe print-failure detection
EOF
    fi
    if [[ -f "$MOONRAKER_ASVC" ]] && ! grep -qx "$SERVICE" "$MOONRAKER_ASVC"; then
        echo "$SERVICE" >> "$MOONRAKER_ASVC"
    fi
    sudo_run systemctl restart moonraker || warn "Restart Moonraker to see Scythe in the Update panel."
    echo "Scythe now updates from the Update panel in Mainsail or Fluidd."
fi

# Wait for the dashboard to answer.
for _ in $(seq 30); do
    curl -fs "http://127.0.0.1:$PORT/api/healthz" >/dev/null 2>&1 && break
    sleep 1
done

step "Done"
bold "Open the setup in your browser:"
for ip in $(hostname -I 2>/dev/null); do
    [[ "$ip" == *:* ]] && continue  # skip IPv6
    echo "  http://$ip:$PORT/"
done
echo "  http://$(hostname).local:$PORT/"
if ! curl -fs "http://127.0.0.1:$PORT/api/healthz" >/dev/null 2>&1; then
    warn "The dashboard isn't answering yet. Check: journalctl -u $SERVICE -n 50"
fi
