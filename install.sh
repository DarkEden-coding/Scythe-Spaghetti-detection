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

# --------------------------------------------------------------------------- #
# Progress output
# --------------------------------------------------------------------------- #

STEP=0
TOTAL_STEPS=0
LOG="$(mktemp -t scythe-install.XXXXXX.log)"
TTY=0
[[ -t 1 ]] && TTY=1
SPINNER='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
trap '((TTY)) && printf "\033[?25h"; true' EXIT

# bar DONE TOTAL WIDTH: a bar like ██████░░░░.
bar() {
    local filled=0 out="" i
    (($2 > 0)) && filled=$(($1 * $3 / $2))
    ((filled > $3)) && filled=$3
    for ((i = 0; i < $3; i++)); do
        if ((i < filled)); then out+="█"; else out+="░"; fi
    done
    printf '%s' "$out"
}

# A numbered step header with the overall progress bar.
step() {
    if ((TOTAL_STEPS)); then
        STEP=$((STEP + 1))
        printf '\n\033[1;33m[%s] %d/%d\033[0m \033[1m%s\033[0m\n' \
            "$(bar $((STEP - 1)) "$TOTAL_STEPS" 20)" "$STEP" "$TOTAL_STEPS" "$*"
    else
        printf '\n\033[1;33m==>\033[0m \033[1m%s\033[0m\n' "$*"
    fi
}

done_line() { printf '  \033[32m✓\033[0m %s\n' "$*"; }

# run_task LABEL CMD...: run CMD with its output going to $LOG, and show a
# spinner with the elapsed time so a slow step never looks stalled. If
# PROGRESS_FN names a function, its output (e.g. "42% 73/175 MB") is shown
# next to the label. On failure, print the end of the log and exit.
run_task() {
    local label="$1" pid start elapsed frame=0 extra last_note=0 log_start
    shift
    printf '\n--- %s\n' "$label" >>"$LOG"
    log_start=$(wc -l <"$LOG")
    "$@" >>"$LOG" 2>&1 </dev/null &
    pid=$!
    start=$SECONDS
    if ((TTY)); then printf '\033[?25l'; else printf '  %s...\n' "$label"; fi
    while kill -0 "$pid" 2>/dev/null; do
        elapsed=$((SECONDS - start))
        extra=""
        if [[ -n "${PROGRESS_FN:-}" ]]; then extra="$("$PROGRESS_FN" 2>/dev/null || true)"; fi
        if ((TTY)); then
            printf '\r\033[K  \033[36m%s\033[0m %s %s\033[2m(%ds)\033[0m' \
                "${SPINNER:frame++%${#SPINNER}:1}" "$label" "${extra:+$extra }" "$elapsed"
        elif ((elapsed - last_note >= 15)); then
            last_note=$elapsed
            printf '  ... still working (%ds) %s\n' "$elapsed" "$extra"
        fi
        # Keep the sudo timestamp fresh through long steps.
        sudo -n -v 2>/dev/null || true
        sleep 0.2
    done
    elapsed=$((SECONDS - start))
    if ((TTY)); then printf '\r\033[K\033[?25h'; fi
    if wait "$pid"; then
        printf '  \033[32m✓\033[0m %s \033[2m(%ds)\033[0m\n' "$label" "$elapsed"
    else
        printf '  \033[31m✗\033[0m %s \033[2m(%ds)\033[0m\n' "$label" "$elapsed"
        echo "Last lines of the log ($LOG):" >&2
        tail -n +"$((log_start + 1))" "$LOG" | tail -n 25 | sed 's/^/    /' >&2
        exit 1
    fi
}

# Ask for the sudo password now if needed; a prompt hidden behind a spinner
# would look like a stall.
ensure_sudo() {
    sudo -n true 2>/dev/null || sudo_run -v
}

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

case "$(uname -m)" in
    aarch64|x86_64|amd64) LOCAL_MODEL=1 ;;
    *) LOCAL_MODEL=0
       warn "This is a 32-bit system, so the local model can't run here. Scythe will use an AI decision model (OpenRouter) instead." ;;
esac
ON_KLIPPER=0
[[ -f "$MOONRAKER_CONF" ]] && ON_KLIPPER=1
TOTAL_STEPS=$((6 + LOCAL_MODEL + ON_KLIPPER))
ensure_sudo

step "Checking prerequisites"
missing=()
for tool in git curl; do
    command -v "$tool" >/dev/null || missing+=("$tool")
done
if ((${#missing[@]})); then
    command -v apt-get >/dev/null || die "Please install: ${missing[*]}"
    run_task "Updating package lists" sudo apt-get update -qq
    run_task "Installing ${missing[*]}" sudo apt-get install -y -qq "${missing[@]}"
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    run_task "Installing uv (Python package manager)" \
        sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi
command -v uv >/dev/null || die "uv did not install. See https://docs.astral.sh/uv/."
done_line "git, curl and uv are ready"

# Write the newest vX.Y.Z tag (or nothing) to the file $1.
latest_release() {
    git ls-remote --tags --refs "$REPO" 'v*' | sed 's#.*refs/tags/##' \
        | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1 >"$1" || true
}

step "Downloading Scythe"
if [[ -z "$REF" ]]; then
    ref_file="$(mktemp)"
    run_task "Finding the newest release" latest_release "$ref_file"
    REF="$(cat "$ref_file")"
    rm -f "$ref_file"
    if [[ -z "$REF" ]]; then
        warn "No release has been published yet; installing the $PRIMARY_BRANCH branch."
        REF="$PRIMARY_BRANCH"
    fi
fi

if [[ -d "$DIR/.git" ]]; then
    if [[ -n "$(git -C "$DIR" status --porcelain --untracked-files=no)" ]]; then
        die "$DIR has local changes. Commit or discard them, then run the installer again."
    fi
    run_task "Fetching updates" git -C "$DIR" fetch --tags --force origin
else
    [[ ! -e "$DIR" ]] || die "$DIR exists but isn't a Scythe checkout. Choose another with --dir."
    run_task "Cloning the repository" git clone --progress "$REPO" "$DIR"
fi

cd "$DIR"
if git rev-parse --quiet --verify "refs/tags/$REF" >/dev/null; then
    # Keep the primary branch checked out at the release tag; Moonraker's
    # update manager expects a branch, and moves it to newer tags.
    git checkout --quiet -B "$PRIMARY_BRANCH" "refs/tags/$REF"
    git branch --quiet --set-upstream-to "origin/$PRIMARY_BRANCH" "$PRIMARY_BRANCH"
    done_line "Release $REF"
elif git rev-parse --quiet --verify "refs/remotes/origin/$REF" >/dev/null; then
    git checkout --quiet -B "$REF" "origin/$REF"
    git branch --quiet --set-upstream-to "origin/$REF" "$REF"
    done_line "Branch $REF ($(git rev-parse --short HEAD))"
else
    die "No branch or tag called $REF."
fi

step "Installing Python and dependencies"
if [[ ! -x .venv/bin/python ]]; then
    run_task "Setting up Python 3.11" uv venv --seed --python 3.11 .venv
fi
# Moonraker's update manager installs requirements with the venv's own pip.
run_task "Installing dependencies (the slowest step on a Pi)" \
    uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
    pip -r requirements.txt

if ((LOCAL_MODEL)); then
    step "Downloading the detection model (175 MB)"
    MODEL_PATH="$(.venv/bin/python -c 'from src.config import load_settings; print(load_settings(validate=False).detection.model_path)')"
    MODEL_URL="$(.venv/bin/python -c 'from src.detection.model_download import MODEL_URL; print(MODEL_URL)')"
    MODEL_BYTES="$(curl -fsSIL "$MODEL_URL" 2>/dev/null | tr -d '\r' \
        | awk 'tolower($1) == "content-length:" { n = $2 } END { print n + 0 }')"
    model_progress() {
        local have
        have="$(stat -c %s "$MODEL_PATH.part" 2>/dev/null || stat -c %s "$MODEL_PATH" 2>/dev/null || echo 0)"
        if ((MODEL_BYTES > 0)); then
            printf '%s %3d%% %d/%d MB' "$(bar "$have" "$MODEL_BYTES" 15)" \
                $((have * 100 / MODEL_BYTES)) $((have >> 20)) $((MODEL_BYTES >> 20))
        else
            printf '%d MB' $((have >> 20))
        fi
    }
    PROGRESS_FN=model_progress run_task "Downloading" .venv/bin/python -m src download-model
fi

step "Configuring Scythe"

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
done_line "Dashboard port $PORT"

step "Installing the service"
ensure_sudo
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
done_line "Service installed"

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
        # The file may lack a final newline; don't glue onto the last entry.
        [[ -z "$(tail -c 1 "$MOONRAKER_ASVC")" ]] || echo >> "$MOONRAKER_ASVC"
        echo "$SERVICE" >> "$MOONRAKER_ASVC"
    fi
    ensure_sudo
    ( run_task "Restarting Moonraker" sudo systemctl restart moonraker ) \
        || warn "Restart Moonraker to see Scythe in the Update panel."
    done_line "Scythe now updates from the Update panel in Mainsail or Fluidd"
fi

# Wait up to 30 s for the dashboard. Never fails: a warning below covers it.
wait_for_dashboard() {
    for _ in $(seq 30); do
        curl -fs "http://127.0.0.1:$PORT/api/healthz" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 0
}

step "Starting Scythe"
ensure_sudo
run_task "Starting the service" sudo systemctl restart "$SERVICE"
run_task "Waiting for the dashboard" wait_for_dashboard
rm -f "$LOG"

printf '\n\033[1;32m[%s] Done\033[0m\n' "$(bar 1 1 20)"
bold "Open the setup in your browser:"
for ip in $(hostname -I 2>/dev/null); do
    [[ "$ip" == *:* ]] && continue  # skip IPv6
    echo "  http://$ip:$PORT/"
done
echo "  http://$(hostname).local:$PORT/"
if ! curl -fs "http://127.0.0.1:$PORT/api/healthz" >/dev/null 2>&1; then
    warn "The dashboard isn't answering yet. Check: journalctl -u $SERVICE -n 50"
fi
