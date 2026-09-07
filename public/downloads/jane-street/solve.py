#!/usr/bin/env python3
"""Recover the target, search English word pairs, and run Jane Street's model.

Python 3.10 is required by the original model's cloudpickled input wrapper.
See README.md. The answer and target are deliberately not embedded here.
"""
import argparse
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from wordfreq import top_n_list

MODEL_SHA256 = "1ff10e21b54431a0959d8d6827d670fe122490c041ac234627fd37f44d825913"
MODEL_URL = "https://huggingface.co/jane-street/2025-03-10/resolve/1a904e632c4224100e840445b0f00aa91e697edb/model.pt"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(data):
    return hashlib.md5(data, usedforsecurity=False).digest()


def recover_lock(model):
    linears = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
    comparator, final = linears[-2:]
    assert comparator.weight.shape == (48, 192)
    weights = comparator.weight.detach().reshape(3, 16, 192)
    biases = comparator.bias.detach().reshape(3, 16)
    assert torch.equal(weights[0], weights[1])
    assert torch.equal(weights[1], weights[2])
    assert torch.equal(biases[0] + 1, biases[1])
    assert torch.equal(biases[1] + 1, biases[2])
    assert final.weight.tolist() == [[1.0] * 16 + [-2.0] * 16 + [1.0] * 16]
    assert final.bias.item() == -15
    assert isinstance(model[-3], torch.nn.ReLU)
    assert isinstance(model[-1], torch.nn.ReLU)
    target = -biases[1]
    assert torch.equal(target, target.round())
    return comparator, weights[0], bytes(int(x) for x in target), len(linears)


def find_phrase(target, words):
    started = time.perf_counter()
    count = 0
    encoded = [word.encode("ascii") for word in words]
    for first in encoded:
        prefix = hashlib.md5(first + b" ", usedforsecurity=False)
        for second in encoded:
            # Search only the short-input region tested against standard MD5.
            if len(first) + 1 + len(second) > 31:
                continue
            digest = prefix.copy()
            digest.update(second)
            count += 1
            if digest.digest() == target:
                return (first + b" " + second).decode("ascii"), count, time.perf_counter() - started
    raise RuntimeError(f"No match in {count:,} pairs; enlarge or change the dictionary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--words", type=int, default=10000,
                        help="Take this many frequency-ranked entries before filtering")
    args = parser.parse_args()
    if platform.python_version_tuple()[:2] != ("3", "10"):
        parser.error("Use Python 3.10 for the original serialized wrapper")
    actual_sha = sha256(args.model)
    if actual_sha != MODEL_SHA256:
        raise ValueError("Model checksum mismatch; refusing to deserialize")
    torch.set_num_threads(1)
    # Explicitly trust only the checksum-pinned official artifact. This pickle
    # contains executable Python; never substitute an arbitrary model here.
    model = torch.load(args.model, map_location="cpu", weights_only=False, mmap=True)
    model.eval()
    comparator, projection, target, linear_count = recover_lock(model)
    print(f"Recovered target: {target.hex()}", flush=True)
    captured = []

    def capture(_module, inputs):
        # The comparator receives 192 activations, not 16 ready-made bytes.
        # Its repeated weight block projects these onto the sixteen digest bytes.
        captured.append((projection @ inputs[0]).detach().clone())

    handle = comparator.register_forward_pre_hook(capture)

    def evaluate(phrase):
        captured.clear()
        with torch.inference_mode():
            output = float(model(phrase).item())
        assert len(captured) == 1
        values = captured[0]
        assert torch.equal(values, values.round())
        network_digest = bytes(int(x) for x in values)
        reference = md5(phrase.encode("ascii"))
        return dict(input=phrase, length=len(phrase), output=output,
                    network_digest=network_digest.hex(), hashlib_md5=reference.hex(),
                    digest_matches_md5=network_digest == reference)

    try:
        # Establish the MD5 hypothesis using non-answer inputs before searching.
        probes = [evaluate(text) for text in ("", "a", "abc", "hello world", "vegetable dog")]
        assert all(row["digest_matches_md5"] and row["output"] == 0 for row in probes)
        words = [word for word in top_n_list("en", args.words) if re.fullmatch("[a-z]+", word)]
        dictionary_sha = hashlib.sha256(("\n".join(words) + "\n").encode()).hexdigest()
        phrase, count, seconds = find_phrase(target, words)
        print(f"Found {phrase!r} after {count:,} hashes ({seconds:.2f}s)", flush=True)
        checks = [evaluate(text) for text in
                  (phrase, phrase.title(), phrase + " ", phrase + "\n", phrase.replace(" ", "  "))]
        assert checks[0]["output"] == 1
        assert all(row["output"] == 0 for row in checks[1:])
        assert all(row["digest_matches_md5"] for row in checks)
        boundaries = [evaluate("a" * size) for size in (30, 31, 32, 33, 55)]
        assert all(row["digest_matches_md5"] for row in boundaries[:2])
        assert all(not row["digest_matches_md5"] for row in boundaries[2:])
    finally:
        handle.remove()

    report = dict(
        run_at=datetime.now(timezone.utc).isoformat(),
        provenance="Fresh implementation of the published approach; answer was known beforehand.",
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         platform=platform.platform(), wordfreq="3.1.1", torch_threads=1),
        model=dict(url=MODEL_URL, sha256=actual_sha, bytes=args.model.stat().st_size,
                   layers=len(model), linear_layers=linear_count),
        target=target.hex(),
        search=dict(source="wordfreq.top_n_list('en', n), filtered to [a-z]+, original frequency order",
                    requested_entries=args.words, filtered_words=len(words),
                    dictionary_sha256=dictionary_sha, max_phrase_bytes=31,
                    attempted_pairs=count, seconds=seconds, phrase=phrase),
        identification_probes=probes, solution_checks=checks, length_boundary_checks=boundaries,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"All checks passed. Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
