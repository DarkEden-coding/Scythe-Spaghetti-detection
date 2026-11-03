# Remote access

By default Scythe answers on your local network at `http://<machine>:8090/`. Don't port-forward it to the internet. Use Tailscale instead.

## Tailscale

[Tailscale](https://tailscale.com) puts your devices on a private network that works from anywhere, with no ports opened. It's free for personal use.

In the setup wizard (or **Settings → Printer, remote access and password…**), the **Remote access** step does it in three clicks:

1. **Install Tailscale** runs Tailscale's official install script.
2. **Log in to Tailscale** opens the login page; sign in and the page updates itself.
3. **Turn on HTTPS address** runs `tailscale serve`, giving Scythe an address like `https://printer.your-tailnet.ts.net/`. Discord alerts link to it.

Install Tailscale on your phone or laptop too, signed in to the same account, and that address works wherever you are.

The HTTPS address needs HTTPS certificates turned on for your tailnet. If the button reports an error about certificates, enable **HTTPS** under **DNS** in the [Tailscale admin console](https://login.tailscale.com/admin/dns) and click it again.

### Doing it by hand

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo tailscale serve --bg --https=443 http://127.0.0.1:8090
```

Then set `web.public_url` in `settings.json` to the address `tailscale serve status` prints, so Discord alerts link to it.

## Mainsail sidebar

**Add to sidebar** in the same step makes Scythe open at `/scythe/` on your printer's address, with a Scythe entry in Mainsail's sidebar. Mainsail has no plugin system, so this changes two files that Mainsail updates don't touch:

- a `location /scythe/` block in `/etc/nginx/sites-available/mainsail` (or `fluidd`), marked with `# scythe-begin` and `# scythe-end`. The previous file is kept as `mainsail.bak-scythe`.
- a link in `~/printer_data/config/.theme/navi.json`, which Mainsail reads. On Fluidd, open `/scythe/` directly.

With the link in place, the Tailscale HTTPS address serves Mainsail, and Scythe is at `https://printer.your-tailnet.ts.net/scythe/`.

### Doing it by hand

Add this inside the `server { ... }` block of `/etc/nginx/sites-available/mainsail`, then run `sudo systemctl reload nginx`:

```nginx
location /scythe/ {
    proxy_pass http://127.0.0.1:8090/;
    proxy_buffering off;
}
```

Then create `~/printer_data/config/.theme/navi.json`:

```json
[{"title": "Scythe", "href": "/scythe/", "target": "_self", "position": 85}]
```

## How the buttons get permission

Scythe runs as your normal user. The installer copies one script, `/usr/local/lib/scythe/scythe-helper`, owned by root, and adds a sudoers rule (`/etc/sudoers.d/scythe`) that lets your user run that script and nothing else. The script has six fixed actions (install Tailscale, log in, turn the HTTPS address on or off, add or remove the nginx block) and accepts no input except a port number. If you'd rather not allow this, delete `/etc/sudoers.d/scythe`; the buttons turn off and the manual steps above still work.

## Password

Anyone who can open the dashboard can pause your printer. On a home network or your tailnet that's usually fine. If others share the network, set a password in the setup's **Password** step. Logins last 30 days, and changing the password logs out every other browser.
