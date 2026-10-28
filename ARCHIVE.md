# ECG runtime experimental archive

Archived on 28 October 2026 before aligning the fork with William's Eko.

| Reference | Purpose |
|---|---|
| `main` | Follow `willxxy/eko` without local implementation changes. |
| `experimental/ecg-core` | Preserve and optionally continue our ECG runtime experiments. |
| `ecg-core` | Retain the existing published branch for old links and checkouts. |
| `archive/ecg-runtime-2026-10-28` | Exact pre-sync snapshot at `89643cbf55715b3af4e0e6687815eb81b502d233`, including the latest README edit. |

The experimental branch adds archive documentation to that snapshot. Its runtime,
scientific tools, evaluation code and saved evidence are preserved.

## Recorded state

The snapshot contains live measurements, the optional 150-label ECGFounder
backend, bounded named-tool execution, evidence-based reporting and the reporting
ablation study. The latest live confirmation completed 17/20, 20/20 and 16/20 full
tasks. The target of at least 18/20 in each repeat remains unmet. These are
reporting-completion results; clinical accuracy is unvalidated.

The one-correction prototype is unit-tested and has no live screen on the current
reporter. Research execution is paused. The saved automated verification reports
199 tests, with 31 optional-dependency skips in core-only execution. Historical
verification and benchmark artifacts identify the source they checked; this
archive operation adds documentation only.

Local records, checkpoints, logs, the API key and the existing virtual environment
remain outside committed source. This checkout's local Git exclusions protect
those artifacts when switching to upstream's `main`.

## Following upstream

The alignment target is
[`b517df0`](https://github.com/willxxy/eko/tree/b517df076cd9bf307410fd04e763476a257923c3).
The remotes have separate roles: `upstream` reads William's repository; `origin`
publishes this fork. Updating the fork does not modify William's repository.

Future updates from a clean checkout:

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
```

Return to the experiments with:

```bash
git switch experimental/ecg-core
```

Port selected ideas into the experimental branch separately. Do not merge the
experimental runtime into the tracking branch. SGLang installation and live
execution of upstream's current code have not been validated by this archive
operation.
