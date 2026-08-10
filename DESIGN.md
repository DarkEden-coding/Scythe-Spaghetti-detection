---
name: Scythe Machine Console
description: A restrained local operator console for 3D-printer failure detection.
colors:
  inspection-black: "#090b0d"
  machine-plane: "#101316"
  raised-graphite: "#15191d"
  control-graphite: "#1a1e22"
  structural-seam: "#2b3036"
  instrument-white: "#f0f1f2"
  telemetry-muted: "#a0a6ad"
  signal-amber: "#ffbd22"
  healthy-green: "#55d879"
  failure-red: "#ff6464"
typography:
  title:
    fontFamily: "ui-sans-serif, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 700
    lineHeight: 1.25
  body:
    fontFamily: "ui-sans-serif, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "ui-sans-serif, system-ui, sans-serif"
    fontSize: "11px"
    fontWeight: 700
    lineHeight: 1.45
    letterSpacing: "0.07em"
rounded:
  compact: "7px"
  control: "8px"
  panel: "12px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "24px"
components:
  button-primary:
    backgroundColor: "{colors.signal-amber}"
    textColor: "{colors.inspection-black}"
    rounded: "{rounded.control}"
    padding: "11px 14px"
    height: "50px"
  panel:
    backgroundColor: "{colors.machine-plane}"
    textColor: "{colors.instrument-white}"
    rounded: "{rounded.panel}"
    padding: "16px"
---

# Design System: Scythe Machine Console

## Overview

**Creative North Star: "The Inspection Bay"**

Scythe feels like a purpose-built inspection station beside a working machine: dark enough for a workshop, precise enough to trust at a glance, and quiet until operator attention is required. Deep charcoal planes, fine structural seams, and one amber signal color create the machine-console world without copying a familiar printer-control product.

The camera is the instrument, not decoration. Information wraps around it as compact telemetry rather than a dashboard of equal cards. Motion communicates the deliberately slow monitoring cadence and state changes; it never simulates a live feed.

**Key Characteristics:**
- Camera-dominant, asymmetric operator layout
- Restrained charcoal and amber material language
- Dense measurement details with quiet space around the inspection image
- Explicit machine states using text, icon, and color
- Crisp, low-motion interactions for continuous use

## Colors

Layered near-black neutrals carry the surface. Signal Amber is reserved for detection geometry, focus, live cadence, and the currently available primary action. Healthy Green confirms an online or clear state; Failure Red appears only for detections, errors, and failed controls.

**The Signal Discipline Rule.** Amber is operational signal, not ambient decoration; a screen covered in accent color has lost its hierarchy.

## Typography

The workhorse system sans stack carries all interface language. Native tabular numerals align timestamps, confidence, areas, and durations; system monospace appears only in the camera frame stamp.

**The Instrument Type Rule.** Monospace belongs only to measurements that benefit from fixed-width comparison; labels and prose remain sans serif.

## Layout

Desktop uses the approved machine-state-spine composition: a 220px left spine, flexible dominant camera center, 306px detection rail right, and a full-width operator deck beneath the workspace. Spacing follows an 8px base rhythm. At 900px the spine becomes a compact horizontal header and the content stacks camera, detection, then controls. At 600px telemetry and actions use two- or one-column layouts with at least 50px controls.

## Elevation & Depth

Depth comes primarily from tonal layering and fine graphite seams. The main operational surfaces use one soft, downward ambient shadow (`0 18px 45px rgba(0, 0, 0, 0.32)`). A component uses either a structural border or strong shadow emphasis, never both as competing decoration.

## Shapes

Panels use machined 12px corners, controls use 8px corners, and compact tools use 7px corners. Pills are limited to tiny counts and state indicators. Detection rectangles remain square and crisp because they are measurement geometry, not containers.

## Components

### Buttons

Controls are 50px minimum height with a quiet graphite default. The available primary action becomes solid amber with dark text; unavailable machine actions stay visible but disabled. Focus always uses a 3px amber outline outside the component.

### Cards / Containers

The camera, detection rail, and operator deck are responsibility-sized surfaces rather than a grid of equal cards. Internal sections are separated by fine seams and use 16–20px padding.

### Signature Component

The camera stage combines a real JPEG, a coordinate-matched SVG overlay, a compact frame timestamp, and a linear cadence track. Detection boxes use amber strokes and solid amber confidence tags, scaling from source-image coordinates without rasterizing labels into the frame.

## Do's and Don'ts

### Do:
- **Do** let the latest camera frame dominate the first viewport.
- **Do** show the slow update interval through timestamp, countdown, and linear cadence.
- **Do** keep acknowledgment distinct from pause and resume.
- **Do** preserve visible keyboard focus and text labels for every state.

### Don't:
- **Don't** imitate the reference screenshot's pipeline editor or operation-list structure.
- **Don't** turn every metric into an equal rounded card.
- **Don't** imply the camera is live video.
- **Don't** use decorative neon glow, gradient text, or monospace as a machine costume.
