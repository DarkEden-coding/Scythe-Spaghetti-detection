# Troubleshooting

Start with `python -m src check` in the install directory (`cd ~/scythe && .venv/bin/python -m src check`). It checks the config, the model, the print state and the camera, and says which one is unhappy.

**The installer finished but the page doesn't load.** Run `journalctl -u scythe -n 50`. If another program took the port, run the installer again with `--port 8100`.

**The wizard can't find my printer.** Enter the address you use for Mainsail or Fluidd, for example `http://mainsailos.local/`, and click **Search**. If Scythe runs on a different machine from Klipper, make sure that machine can open the address too.

**"Webcam 'Bed' not found".** The camera name is case sensitive. Re-run the setup from Settings and pick the camera from the list.

**The Tailscale and Mainsail buttons are greyed out.** Scythe isn't allowed to run its helper. Run the install command again, which adds the sudoers rule, or follow the manual steps in [Remote access](remote-access.md).

**I forgot the dashboard password.** Delete the `password_hash` line from `~/scythe/settings.json` and run `sudo systemctl restart scythe`.

**Scythe doesn't show in Mainsail's Update panel.** Check that `~/printer_data/moonraker.asvc` contains `scythe` and `moonraker.conf` has an `[update_manager scythe]` section, then restart Moonraker. Running the install command again adds both.

**Too many Discord messages.** Set **Status messages** to *Keep one message updated* or *Alerts only*.

**Permission errors.** Don't run Scythe with `sudo`. Give your user the install directory back: `sudo chown -R "$USER" ~/scythe`.
