# Discord

Discord is optional. Without it, alerts appear on the dashboard only.

## Setting it up

The setup wizard and **Settings → Discord → How do I set this up?** both open a four-step guide with pictures. In short:

1. In the [Discord Developer Portal](https://discord.com/developers/applications), create an application, open **Bot**, click **Reset Token** and copy the token.
2. Paste the token into Scythe. The guide then shows an invite link with the right permissions; open it and pick your server.
3. In Discord, turn on **User Settings → Advanced → Developer Mode**.
4. Right-click the alert channel and choose **Copy Channel ID**, then right-click your name and choose **Copy User ID**. Paste both into Scythe.

Scythe connects to Discord at startup, so after changing these in Settings, click **Restart now** on the banner that appears.

Treat the token like a password. Scythe stores it in `settings.json` (readable only by your user) and never sends it back to the browser.

## What it posts

- **Failure alerts** ping you with the frame and the detection boxes or scores. If you set up a Tailscale address, the alert title links to the dashboard. React with 👍 to dismiss the alert and resume monitoring.
- **Status updates.** *Keep one message updated* edits a single message with the latest frame. *Post a new one each frame* posts every time. *Alerts only* posts nothing else.

## Slash commands

Only the user whose ID you entered can use these:

- `/pause` pauses the print.
- `/resume` resumes a paused print.
- `/get_printer_status` reports the print state.
- `/get_image` sends the current camera frame.
- `/get_log_file` sends Scythe's log.
