# Architecture

## Layout

```
main.py, settings_ui.py     Thin shims for the old entry points.
src/
  cli.py                    Argument parsing; run / configure / check / env / update.
  app.py                    Composition root — the only place that picks implementations.
  config.py                 Typed settings: defaults -> settings.json -> environment.
  paths.py                  Every filesystem location, resolved from the install root.
  errors.py                 Exception hierarchy.
  events.py                 What the monitor emits.
  monitor.py                The loop. Knows nothing about Discord or Moonraker.
  updater.py                Optional git fast-forward self-update.
  logging_setup.py          Console + rotating file handlers.
  detection/
    detector.py             SpaghettiDetector — lazy model load, returns value objects.
    results.py              DetectionBox / DetectionResult.
    annotate.py             draw_boxes(image, boxes) -> Image. Pure PIL.
    export.py               .pt -> .onnx.
  printer/
    base.py                 PrinterClient protocol, PrintState enum.
    moonraker.py            HTTP client with session reuse, timeouts, retries.
  notify/
    base.py                 Notifier / Acknowledgement protocols, Null + Composite.
    discord_notifier.py     Discord implementation.
  utils/                    Formatting and image helpers.
tools/                      Dev-only: train, export, dataset collection.
models/                     Weights.
data/                       Runtime output: logs, annotated frames. Gitignored.
tests/
```

## The dependency rule

Dependencies point inward:

```
cli -> app -> monitor -> {PrinterClient, SpaghettiDetector, Notifier}
                              ^              ^              ^
                        moonraker.py    detector.py   discord_notifier.py
```

`monitor.py` imports the protocols in `printer/base.py` and `notify/base.py`, never
the concrete modules. `app.py` is the single place that decides Moonraker and
Discord are the implementations in use. That is what makes the roadmap additive.

## Adding things

**A notification channel (email, ntfy, a web UI).** Implement `Notifier` in
`src/notify/`. Return an `Acknowledgement` from `notify()` if your channel can
carry a "I've seen it" signal; return `None` if it can't. Then add it to the list in
`app.build_notifier` — `CompositeNotifier` fans out to all of them and survives one
being down.

**A printer backend (OctoPrint, Duet).** Implement `PrinterClient` in
`src/printer/`, mapping the firmware's state vocabulary onto `PrintState`.
Methods are synchronous; the monitor dispatches them to worker threads.

**A setting.** Add the field to the right dataclass in `config.py`. It picks up JSON
loading, type coercion, an environment variable, and serialisation automatically.
Add validation to that class's `validate()`, and a `Prompt` in `wizard.py` if a user
should be asked about it.

**An event.** Add a frozen dataclass to `events.py` and a handler entry in each
notifier's dispatch table. Notifiers ignore events they have no handler for, so
partial support is fine.

## Rules the code follows

**No import-time I/O.** Importing a module never touches the network, loads a
model, or reads config. `MoonrakerClient.__init__` does no HTTP; the webcam URL is
resolved on first use. `SpaghettiDetector.__init__` loads nothing; `load()` does.
This is what makes the test suite runnable without a printer, a model, or Torch.

**No implicit filesystem coupling.** The detector returns a `DetectionResult`
carrying the frame; the caller decides what to persist and send. Modules don't
communicate by writing and re-reading files.

**No CWD dependence.** Paths come from `paths.py`, derived from the package
location. Relative paths in config resolve against the install root.

**The loop survives everything.** Any exception from a tick is caught, reported,
and the loop continues. Repeat failures are throttled so an unplugged printer
doesn't flood the channel.

**Nothing blocking runs on the event loop.** HTTP and inference go through
`asyncio.to_thread`. Inference on a Pi takes seconds; blocking the loop drops the
Discord heartbeat.

## Testing

The suite needs no printer, model, token, or network — and deliberately does not
require `ultralytics`, so it stays fast in CI. `tests/conftest.py` has `FakePrinter`,
`FakeDetector`, and `RecordingNotifier`; `tests/test_moonraker.py` has a fake
`requests.Session`; `tests/test_detection.py` has a fake Ultralytics model.
