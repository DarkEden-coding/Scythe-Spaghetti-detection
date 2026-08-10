# Scythe Spaghetti Detection

Hello, thanks for checking out my project! This is a **simple and lightweight** *OPEN SOURCE* and *SELF-HOSTED* 3D printer fail detection project. It is entirely based in Python and currently uses the **YOLOv8 object detection framework**. The project currently has a large model trained. If needed, I will try to train models for other sizes suitable for less powerful devices such as the Raspi-Zero. If you find any issues please make an issue or pull request, any showcase of your usage of this project would be greatly appreciated, you can contact me using the info below.

## This project is in alpha! I apologise for any errors or dificulty you have with this project, but as a student I may not have the time to find all the errors, if you find any please report them in the Issues!

## Features:
- [x] **Supports any moonraker supported 3d printer**
- [x] **3D Printer Fail Detection**
- [x] **Discord Notifications and Status Updates**
- [x] **Lightweight and Easy to Use**
- [x] *Open Source*
- [x] *Self Hosted*
- [x] **Linux and Windows Support**
- [x] **YOLOv8 Object Detection**
- [x] *Large Model Trained*
- [x] Any Linux or Windows-based system support (theoretically, not yet tested, should only require Python support)
- [ ] XL Model Trained
- [ ] Medium Model Trained
- [ ] Small Model Trained
- [ ] Nano Model Trained
- [x] **Local Web Interface**
- [ ] Email Notifications

## Slash Commands
Only the configured `discord.ping_user_id` can use these commands:

- `/pause` pauses an active print.
- `/resume` resumes a paused print; other printer states are rejected.
- `/get_printer_status` reports the current Moonraker print state.
- `/get_image` returns the current camera frame.
- `/get_log_file` returns the current Scythe log for troubleshooting.

## Installation:
Install on the machine that can reach your printer.

1. Clone the repository
```bash
git clone "https://github.com/DarkEden-coding/Scythe-Spaghetti-detection.git"
```
2. Move into the installation folder
```bash
cd Scythe-Spaghetti-detection
```
3. Install the required packages
```bash
pip install -r requirements.txt
```

Installing as a package also gives you the `scythe` command:
```bash
pip install -e .
```

## Configuration:
1. Run the configuration wizard:
```bash
python3 -m src configure
```
Follow the on screen instructions. Press Enter at any prompt to keep the value in brackets.

2. To get the bot token, follow this guide: https://discordpy.readthedocs.io/en/stable/discord.html
3. To get the discord channel id and user id enable developer mode (settings -> advanced -> dev mode) in discord and right click the channel/user and click "copy id"

Settings are written to `settings.json`. **That file contains your bot token — it is gitignored, keep it that way.**

Check everything before you rely on it:
```bash
python3 -m src check
```
This validates the config, reads the print state, resolves the webcam, and pulls a test frame.

### Settings reference

| Setting | Default | What it does |
| --- | --- | --- |
| `discord.bot_token` | — | Your bot token. |
| `discord.log_channel_id` | — | Channel the bot posts to. |
| `discord.ping_user_id` | — | Who gets pinged on a failure. |
| `discord.status_update_mode` | `edit` | `edit` keeps one status message current, `message` posts a new one each loop, `silent` posts none. |
| `discord.acknowledge_timeout` | `0` | Seconds to wait for a 👍 after a failure before resuming. `0` waits forever. |
| `web.enabled` | `true` | Serve the local operator dashboard with monitoring. |
| `web.host` | `0.0.0.0` | Listen on every interface so other devices on the LAN can connect. |
| `web.port` | `8080` | Dashboard HTTP port. |
| `printer.url` | `http://mainsailos.local/` | Moonraker address. |
| `printer.webcam_name` | `Bed` | Exactly as it appears in Mainsail/Fluidd. Case sensitive. |
| `printer.request_timeout` | `10` | Seconds before a request to the printer gives up. |
| `printer.max_retries` | `3` | Retries for reads (never for pause). |
| `detection.use_cuda` | `false` | Use an NVIDIA GPU. |
| `detection.use_onnx` | `true` | Use the ONNX runtime — much faster on a Pi. |
| `detection.min_confidence` | `0.6` | Raise for fewer false positives, lower for fewer misses. |
| `detection.save_annotated_frames` | `true` | Keep failure frames in `data/detections/`. |
| `target_loop_time` | `30` | Seconds between checks. |
| `pause_on_spaghetti` | `true` | Pause the print when a failure is detected. |
| `enable_auto_update` | `false` | Pull updates from git on start. |
| `log_level` | `INFO` | `DEBUG` when reporting a bug. |

Every setting can also be supplied as an environment variable, which is handy for systemd or Docker and keeps your token out of any file:
```bash
python3 -m src env       # lists them all
export SCYTHE_DISCORD_BOT_TOKEN=...
```
Precedence is defaults → `settings.json` → environment.

### Upgrading from an older version
If you already have a `settings.py`, your values are imported automatically the first time you run `scythe configure` (the file is parsed, never executed). **Delete `settings.py` afterwards — it still contains your bot token.**

## Usage:
```bash
python3 -m src run     # or: scythe run, or: python3 main.py
```

### Web dashboard

The dashboard starts with the monitor at `http://<scythe-host>:8080`. It shows
the latest camera frame at the configured detection cadence, draws current
detection boxes, reports the latest alert, and provides state-aware pause,
resume, and acknowledgment controls. The **Idle detection** toggle temporarily
runs inference while the printer is not active for debugging; it resets to off
on restart and never pauses the printer, notifies Discord, or waits for an
acknowledgment.

The dashboard intentionally has no login and binds to the local network by
default. Do not port-forward it or expose it to the public internet. Set
`web.host` to `127.0.0.1` if it will only be used through a local reverse proxy.

The monitor command starts both Discord and the dashboard.

To start on boot, use a systemd service. A ready-made unit is in [`deploy/scythe.service`](deploy/scythe.service):
```bash
sudo cp deploy/scythe.service /etc/systemd/system/
sudo nano /etc/systemd/system/scythe.service   # set User= and WorkingDirectory=
sudo systemctl daemon-reload
sudo systemctl enable --now scythe
```
Then check on it with:
```bash
systemctl status scythe
journalctl -u scythe -f
```
Logs are also written to `data/scythe.log`.

## Discord Message UI:
### status:
![Alt text](https://github.com/DarkEden-coding/Scythe-Spaghetti-detection/blob/master/readme_images/status.png?raw=true "Status Message")

### fail:
![Alt text](https://github.com/DarkEden-coding/Scythe-Spaghetti-detection/blob/master/readme_images/fail.png?raw=true "Fail Message")
## Defenetly 3d printer spaghetti, trust :)
![Alt text](https://github.com/DarkEden-coding/Scythe-Spaghetti-detection/blob/master/readme_images/Screenshot%202024-02-03%20at%201.58.17%20PM.png?raw=true "The Spaghett")

## Trouble Shooting:
Run `python3 -m src check` first — it will usually tell you which part is unhappy.

**"Webcam 'Bed' not found"** — the name is case sensitive and must match Mainsail/Fluidd exactly. The error lists the names your printer reported.

**"No configuration found"** — run `python3 -m src configure`.

**Permission errors** — don't run the bot with `sudo`. Fix ownership of the install directory instead:
```bash
sudo chown -R "$USER" /path/to/Scythe-Spaghetti-detection
```
Running as root was the previous advice here; it makes the auto-updater able to rewrite files it shouldn't and is not needed for anything the bot does.

**Too many Discord messages** — set `discord.status_update_mode` to `edit` (the default) or `silent`.

## Development:
```bash
pip install -r requirements-dev.txt
pytest                                 # the suite needs no printer, model, or token
ruff check src tools tests
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together and where to add a new notification channel or printer backend.

Model tooling lives in `tools/`:
```bash
python3 -m tools.export_onnx models/largeModel.pt
python3 -m tools.train --data path/to/data.yaml --name my-run
python3 -m tools.collect_dataset --interval 5 --output data/training
```

## Contributing:
If you would like to contribute to the project, feel free to make a pull request. I will review it and merge it if it is suitable. If you have any questions, feel free to reach out to me on discord: `scytheeden` or email me at `darkedenc9@gmail.com`. If anyone is willing to "donate" gpu time to train new models that would be greatly appreciated.

# Relevant Links:
- [My Discord](https://discord.gg/users/806281289040396288)
- [YOLOv8](https://docs.ultralytics.com/)
- [Roboflow Dataset](https://universe.roboflow.com/dark-eden-nuheg/3d-printing-fail-detection)

# Notes:
- If you are a large company looking into using this software please contact me through email about possable modification to better suit your location and system.
- I am not responsable for any damage to your printer while using this project, use at your own risk, the detection will be as good as I can get it but if you have a spaghetti that is not detected and damages your printer I am not responsable.
