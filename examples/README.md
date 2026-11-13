# Examples

These examples are **synthetic** — they are hand-written toy samples
for trying out the pipeline, not real research data. No real dataset
is included in this repository.

## Files

- `sample_qa.jsonl` — 2 synthetic chart QA samples. Each plots two or
  three model accuracy curves with matplotlib; the key evidence is the
  Model-A curve in both cases.
- `quickstart.py` — loads the samples, scores their difficulty, builds
  an EP/EA counterfactual pair for each, and prints whether the EP view
  keeps the evidence and the EA view removes it.

## Run

From the repository root:

```bash
python examples/quickstart.py
```

Expected output: both samples load, and the EP/EA checks pass
(EP keeps `Model-A`, EA removes `Model-A`).

You can also feed the examples into the dataset pipeline:

```bash
python scripts/build_dataset.py --input examples/sample_qa.jsonl \
    --output /tmp/example_dataset.jsonl --difficulty-threshold 0.0
```

Note the `--difficulty-threshold 0.0` here: these short toy samples
score low on difficulty by design, so the default 0.7 threshold
would filter them out.
