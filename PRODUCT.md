# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Scythe is operated by people running a Moonraker-compatible 3D printer on their local network. The primary job is to monitor a print for spaghetti failures, understand the latest detection at a glance, and safely pause or resume the printer without opening another control surface.

## Product Purpose

Scythe is a self-hosted Python service that periodically captures the printer camera, runs YOLO-based spaghetti detection, can pause a failed print, and reports status through Discord and a local web dashboard. Success means an operator can notice, inspect, and acknowledge a likely failure before it causes more damage.

## Positioning

Scythe combines the existing Moonraker camera and printer API with local fail detection, synchronized Discord reporting, and a lightweight operator dashboard rather than requiring a hosted monitoring service.

## Operating Context

The service runs continuously on a Windows or Linux machine that can reach Moonraker. Operators check it from another device on the same trusted local network. Camera and detection updates follow the configured monitor interval rather than behaving like a live video stream.

## Capabilities and Constraints

- Supports Moonraker-compatible printers and the configured Moonraker webcam.
- Runs detection only while the normalized printer state is active.
- The web dashboard starts with monitoring on port 8080 and is reachable on the local network.
- The first web release has no login or access token by explicit product decision.
- Discord remains active as a parallel notification channel.
- The dashboard exposes pause, resume, and detection acknowledgment only; no cancel or emergency-stop control.
- Either a dashboard acknowledgment or Discord reaction may release a pending detection.
- The dashboard retains the current frame/status and latest detection only, not a persistent history.
- The web camera updates from the same monitor events and cadence used for Discord status updates.

## Brand Commitments

Keep the Scythe name and its simple, lightweight, open-source, self-hosted character. The operator interface uses a simple, sleek dark machine-console style derived from the supplied charcoal interface with a restrained amber/yellow signal accent; it must remain its own design rather than copy the reference layout.

## Evidence on Hand

- Existing Moonraker, detection, annotation, event, Discord, and printer-control implementations in `src/`.
- Existing status and failure screenshots in `readme_images/`.
- A user-supplied UI inspiration screenshot establishes the desired visual material and color direction.
- No claims about detection accuracy, hardware benchmarks, or commercial deployments should be invented.

## Product Principles

- Make current machine state and detection risk readable at a glance.
- Keep control actions narrow, explicit, and state-aware.
- Reuse the monitor event cadence so the dashboard does not create a second camera or inference loop.
- Stay self-hosted and operationally lightweight.
- Preserve failure evidence without implying certainty the detector cannot provide.

## Accessibility & Inclusion

Controls must be keyboard operable, visibly focused, and labeled beyond color alone. Status, confidence, and failures must remain understandable without relying only on amber, green, or red.
