# Jane Street neural-network puzzle: executable solution

This is a fresh implementation and end-to-end verification of the published
solution, made on September 7, 2026. The answer and MD5 interpretation were known
before this work. It is not a claim of blind discovery or original authorship of
the puzzle. The solver itself contains neither the winning phrase nor its digest.

## Reproduce

Use Python **3.10**: the original model contains a cloudpickled Python 3.10 input
wrapper, which cannot be deserialized directly under Python 3.12. Allow roughly
2 GB of disk space plus several GB of available memory. No GPU is required.

With uv installed, run these commands from the extracted bundle:

```sh
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
curl --fail --location --retry 2 \
  'https://huggingface.co/jane-street/2025-03-10/resolve/1a904e632c4224100e840445b0f00aa91e697edb/model.pt' \
  --output model.pt
.venv/bin/python solve.py --model model.pt --output my-results.json
```

The 1,158,692,162-byte model is not bundled. Before deserialization, the script
requires SHA-256:

`1ff10e21b54431a0959d8d6827d670fe122490c041ac234627fd37f44d825913`

The official model includes executable Python, so `weights_only=False` is
necessary for its original wrapper. The checksum pins the specific Jane Street
artifact used here; do not relax this check to load unknown pickles.

## What the program does

1. Checks the model's final comparator structure and extracts its target from
   the middle group of biases.
2. Hooks the penultimate linear layer, projects its 192 input activations onto
   sixteen bytes using the real weights, and compares them with Python's MD5 on
   five non-answer inputs.
3. Enumerates ordered pairs from wordfreq 3.1.1's first 10,000 English entries,
   retaining only lowercase ASCII words. It preserves frequency order and only
   tests phrases up to 31 bytes. Neither word is specially inserted or promoted.
4. Runs the discovered phrase through the complete original model and requires
   output 1. Requires output 0 for capitalization and whitespace variants.
5. Tests repeated `a` inputs at lengths 30, 31, 32, 33, and 55 to reproduce the
   length-encoding bug. It records both network digests and hashlib digests.
6. Writes measured results, search counts, timings, model identity, and the
   ordered dictionary's checksum to JSON.

`results.json` contains the actual run, including spoilers. Runtime varies by
machine. An exhausted dictionary raises an error; it does not establish that
there is no solution outside that candidate set. The probe checks establish
agreement on those inputs, not equivalence to MD5 for every short string.

The word list is generated from the pinned wordfreq dependency, not redistributed
in this bundle. See https://github.com/rspeer/wordfreq for its data sources and
licenses. The list checksum in results.json makes changes detectable.

## Attribution

- Puzzle, model, and creator's explanation:
  https://blog.janestreet.com/can-you-reverse-engineer-our-neural-network/
- Official model: https://huggingface.co/jane-street/2025-03-10
- Official app/input wrapper: https://huggingface.co/spaces/jane-street/puzzle/blob/main/app.py
- Published analysis cited by the accompanying post:
  https://thomasbrownback.com/20260711.html
  https://medium.com/@shreyasmahimkar/reverse-engineering-a-neural-network-md5-hash-puzzle-848c82ed9285

This concerns the original tensor/hash puzzle, not the later shuffled-layer
challenge. The article's interactive diagrams remain explanatory reconstructions;
the JSON report contains measured model activations and outputs.
