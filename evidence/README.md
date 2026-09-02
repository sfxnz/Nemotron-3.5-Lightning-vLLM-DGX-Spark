# evidence/

Receipts for every number in `README.md`. `recipe.yaml` points each measured row at a file here. `python3 kit/render.py --check` lists the rows that have none.

What belongs here:

- `<unit>-<UTC stamp>/bench.txt` and `bench.json`: output of `python3 kit/bench_decode.py --recipe . --phase both --out evidence/<unit>-<UTC stamp>`, the human log with the `SUMMARY` JSON and the machine rows. One directory per frozen wave.
- `<unit>-<UTC stamp>/<probe>.txt` and `probes.json`: output of `kit/probes/run-all.sh . evidence/<unit>-<UTC stamp>` (request, response, HTTP status, `verdict=`).
- `<unit>-<UTC stamp>/doctor.txt`, `run.log`, `stop.txt`: state of the serve the probes and bench ran against.
- `trail.tsv`, `decision.tsv`: what was tried, kept, reverted, and why.

Commit the raw output. Do not type numbers into a file.
