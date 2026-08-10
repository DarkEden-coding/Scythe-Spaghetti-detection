---
version: 1
slug: "src-web-static-index-html"
primary_target: "src/web/static/index.html"
related_targets: ["src/web/static/style.css","src/web/static/app.js"]
---

# Web operator console

- **Scope / mode:** Root dashboard, Operate.
- **Audience / job:** A single operator on the trusted local network checks the latest printer-camera inspection, understands detections, pauses or resumes the print, and acknowledges a failure.
- **Content / constraints:** Uses the monitor event cadence; current frame and latest detection only; Discord remains parallel; no login by explicit decision; port 8080 on the LAN; never imply live video.
- **Chosen direction:** “The Inspection Bay,” approved comp `.impeccable/mocks/scythe-console-c.png`. A narrow left state spine, dominant center camera, low telemetry/control deck, and a compact right detection rail. The memorable moment is measurement geometry drawn directly over the machine view.
- **Responsive:** Below 900px the spine becomes a compact header, camera stays first, detection rail and operator deck stack beneath it, and buttons retain 48px targets.
- **Unresolved:** None for the first release.

## Implementation fidelity inventory

| Commitment | Medium |
| --- | --- |
| Narrow machine-state spine with product mark, connection, print state, last frame, and next scan | Semantic HTML/CSS with inline SVG icons |
| Dominant camera frame with true image aspect ratio | JPEG from the existing monitor event |
| Crisp amber detection rectangles and confidence tags in source-image coordinates | Semantic SVG overlay |
| Right detection verdict and box list | Semantic HTML, generated from current API state |
| Low operator deck with cadence, metrics, Pause, Resume, and conditional Acknowledge | Semantic HTML/CSS/JS |
| Tonal charcoal material, fine seams, machined corners | CSS variables, borders, and one soft offset shadow |
| Slow update legibility | Text timestamps plus linear cadence bar; no fake streaming motion |
| Responsive translation | One 900px media query; no alternate mobile product hierarchy |
