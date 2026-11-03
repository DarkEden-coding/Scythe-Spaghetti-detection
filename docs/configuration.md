# Configuration

The setup wizard and Settings cover what most people need. Everything else lives in `settings.json` in the install directory (`~/scythe/settings.json`). The file holds your tokens, so Scythe keeps it readable only by your user and git ignores it. Restart after editing it by hand: `sudo systemctl restart scythe`.

Every setting can also come from an environment variable, which keeps secrets out of the file. `python -m src env` lists them all. Precedence is defaults, then `settings.json`, then the environment.

```ini
# sudo systemctl edit scythe
[Service]
Environment=SCYTHE_DISCORD_BOT_TOKEN=...
```

| Setting | Default | What it does |
| --- | --- | --- |
| `printer.url` | `http://localhost/` | Moonraker, or the Mainsail/Fluidd address in front of it. |
| `printer.webcam_name` | `Bed` | Camera name exactly as Mainsail/Fluidd shows it. Case sensitive. |
| `printer.request_timeout` | `10` | Seconds before a request to the printer gives up. |
| `printer.max_retries` | `3` | Retries for reads. Pausing is never retried. |
| `target_loop_time` | `30` | Seconds between camera frames. |
| `pause_on_spaghetti` | `true` | Pause the print when a failure is detected. |
| `detection.backend` | `yolo` | `yolo` for the local model, `decision` for a decision model. |
| `detection.interval` | `0` | Seconds between detections. `0` checks every frame. |
| `detection.min_confidence` | `0.6` | Local model: score a box needs to count. |
| `detection.model_path` | `models/largeModel.onnx` | Local model file. `python -m src download-model` fetches it. |
| `detection.use_cuda` | `false` | Run the local model on an NVIDIA GPU (needs `onnxruntime-gpu`). |
| `detection.save_annotated_frames` | `true` | Keep failure frames in `data/detections/`. |
| `detection.decision.provider` | `openrouter` | `openrouter`, or `ollama` for self-hosted. |
| `detection.decision.api_key` | | OpenRouter key. |
| `detection.decision.model` | `~liquid/d1-ultra-latest` | Decision model id. |
| `detection.decision.threshold` | `0.6` | Flag a failure when either answer reaches this. |
| `detection.decision.crop` | | Crop as `left,top,right,bottom` fractions. Easier to drag in Settings. |
| `detection.decision.base_url` | | Endpoint override. |
| `detection.decision.image_max_side` | `1240` | Longest side of the image sent, in pixels. |
| `detection.decision.request_timeout` | `30` | Seconds before a request gives up. |
| `discord.bot_token` | | Bot token. Empty turns Discord off. |
| `discord.log_channel_id` | | Channel for alerts and status. |
| `discord.ping_user_id` | | Who gets pinged, and who can use slash commands. |
| `discord.status_update_mode` | `edit` | `edit`, `message` or `silent`. See [Discord](discord.md). |
| `discord.acknowledge_timeout` | `0` | Seconds to wait for a 👍 after an alert before resuming. `0` waits forever. |
| `web.enabled` | `true` | Serve the dashboard. |
| `web.host` | `0.0.0.0` | Listen address. `127.0.0.1` limits it to this machine and the Mainsail link. |
| `web.port` | `8090` | Dashboard port. The installer picks a free one. |
| `web.public_url` | | Address Discord alerts link to. Set by the Tailscale step. |
| `web.password_hash` | | Set through the wizard's Password step. Empty means no login. |
| `enable_auto_update` | `false` | Update to new releases on start. The installer turns it on when Moonraker isn't there to do it. |
| `setup_complete` | `false` | Monitoring starts once the setup wizard is finished. |
| `log_level` | `INFO` | `DEBUG` logs every score; use it when reporting a bug. |

Logs go to the journal (`journalctl -u scythe -f`) and to `data/src.log`.
