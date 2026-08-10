"""Automated usability verification for the synthetic degradation generator.

Runs the full C-role deliverable end to end on deterministic fixture images and
asserts every documented behaviour. Self-contained: only needs numpy + Pillow
(deps already in requirements.txt); no pytest required.

Run from the repo root (works whether or not ``pip install -e .`` was run):

    python -m scripts.synthetic.verify

Exit code 0 = every check passed, 1 = at least one check failed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Make ``photo_revival`` importable without an installed package.
try:  # pragma: no cover - environment dependent
    import photo_revival  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np
from PIL import Image

from scripts.synthetic.engine import (
    DegradationConfig,
    coverage_of,
    degrade_target,
    derive_variant_seed,
)
from scripts.synthetic import generate as gen

FIXTURE_NAMES = ["fixture_1", "fixture_2", "fixture_3"]
SEVERITIES = ["light", "medium", "heavy"]


def _args(**overrides) -> SimpleNamespace:
    defaults = dict(
        batch_id=None,
        clean_dir=None,
        output_dir=None,
        manifest=None,
        severities=SEVERITIES,
        min_coverage=1.0,
        max_coverage=35.0,
        fill_mode="paper",
        global_seed=0,
        limit=None,
        sample=None,
        global_only=False,
        no_collages=False,
        force=False,
        contact_sheet_only=False,
        verify_only=False,
        make_fixture=False,
        fixture_dir=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Checks:
    def __init__(self) -> None:
        self._checks: list[tuple[str, str, str]] = []

    def run(self, name: str, fn: object) -> None:
        try:
            detail = fn()
        except SystemExit as exc:
            detail = f"unexpected exit code {exc.code}"
            self._checks.append((name, "FAIL", detail))
            return
        except Exception as exc:  # noqa: BLE001
            self._checks.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
            return
        self._checks.append((name, "PASS", str(detail or "ok")))

    def summary(self) -> tuple[int, int]:
        passed = sum(1 for _, status, _ in self._checks if status == "PASS")
        failed = len(self._checks) - passed
        return passed, failed


def main() -> int:
    checks = Checks()
    with tempfile.TemporaryDirectory(prefix="verify_synthetic_") as tmp:
        tmp = Path(tmp)
        clean_dir = tmp / "clean"
        clean_dir.mkdir()
        out_dir = tmp / "out"
        out_global = tmp / "out_global"
        out_sample = tmp / "out_sample"

        # 01 - engine + CLI import cleanly
        def check_imports() -> str:
            assert callable(degrade_target), "degrade_target not callable"
            assert callable(coverage_of), "coverage_of not callable"
            assert callable(gen.generate) and callable(gen.verify), "CLI functions missing"
            assert gen.build_parser(), "parser not constructible"
            return "engine + CLI importable"

        # 02 - per-variant seed derivation is stable and distinct
        def check_seed_derivation() -> str:
            s1 = derive_variant_seed(0, 0, "light")
            s2 = derive_variant_seed(0, 0, "light")
            s3 = derive_variant_seed(0, 1, "light")
            s4 = derive_variant_seed(0, 0, "heavy")
            assert s1 == s2, "seed derivation not deterministic"
            assert s1 != s3 and s1 != s4, "seed collisions across index/severity"
            return f"seed stable ({s1})"

        # 03 - fixture generation
        def check_fixture() -> str:
            gen.make_fixture(_args(make_fixture=True, fixture_dir=str(clean_dir)))
            names = sorted(p.stem for p in clean_dir.glob("*.png"))
            assert names == FIXTURE_NAMES, f"fixture names {names} != {FIXTURE_NAMES}"
            for path in clean_dir.glob("*.png"):
                with Image.open(path) as image:
                    image.verify()
            return f"{len(FIXTURE_NAMES)} fixture images written"

        # 04 - end-to-end generation (3 sources -> 9 variants)
        def check_generate() -> str:
            args = _args(clean_dir=str(clean_dir), output_dir=str(out_dir), limit=3)
            sources = gen._select_sources(clean_dir, None, 3)
            assert len(sources) == 3
            gen.generate(args, sources, out_dir)
            manifest = gen.read_csv(out_dir / "manifest.csv")
            assert len(manifest) == 9, f"manifest has {len(manifest)} rows, expected 9"
            counts = {sev: sum(1 for r in manifest if r["severity"] == sev) for sev in SEVERITIES}
            assert counts == {"light": 3, "medium": 3, "heavy": 3}, f"severity counts {counts}"
            for sub in ("target", "degraded", "masks", "metadata"):
                assert (out_dir / sub).is_dir(), f"missing {sub}/"
            assert (out_dir / "summary.json").is_file()
            assert (out_dir / "contact_sheet.jpg").is_file()
            assert (out_dir / "collages").is_dir()
            return "9 variants generated with full layout"

        # 05 - generator's own verify passes
        def check_verify() -> str:
            gen.verify(_args(output_dir=str(out_dir)), out_dir)
            return "verify-only PASSED"

        # 06 - determinism: same seed reproduces identical bytes
        def check_determinism() -> str:
            manifest = gen.read_csv(out_dir / "manifest.csv")
            row = next(r for r in manifest if r["severity"] == "medium")
            seed = int(row["seed"])
            with Image.open(out_dir / row["target_path"]) as opened:
                target = opened.convert("RGB")
            cfg = DegradationConfig()
            v1 = degrade_target(target, "medium", seed, cfg)
            v2 = degrade_target(target, "medium", seed, cfg)
            assert np.array_equal(np.asarray(v1.mask), np.asarray(v2.mask)), "mask not reproducible"
            assert np.array_equal(np.asarray(v1.degraded), np.asarray(v2.degraded)), "image not reproducible"
            saved = np.asarray(Image.open(out_dir / row["mask_path"]))
            assert np.array_equal(np.asarray(v1.mask), saved), "regenerated mask != saved mask"
            return "same seed -> identical mask + image, matches saved file"

        # 07 - mask properties across all variants
        def check_mask_properties() -> str:
            manifest = gen.read_csv(out_dir / "manifest.csv")
            for row in manifest:
                mask = Image.open(out_dir / row["mask_path"])
                target = Image.open(out_dir / row["target_path"])
                degraded = Image.open(out_dir / row["degraded_path"])
                assert mask.mode == "L", f"{row['source_id']}/{row['severity']} mask mode {mask.mode}"
                assert mask.size == target.size == degraded.size, f"{row['source_id']}/{row['severity']} size mismatch"
                values = set(np.asarray(mask).ravel().tolist())
                assert values.issubset({0, 255}), f"{row['source_id']}/{row['severity']} non-binary mask {values}"
                if row["restoration_type"] == "local_repair":
                    assert max(values) == 255, f"{row['source_id']}/{row['severity']} empty local-repair mask"
                mask.close(); target.close(); degraded.close()
            return f"{len(manifest)} masks: L mode, binary, size-matched"

        # 08 - coverage within configured bounds and consistent with metadata
        def check_coverage() -> str:
            manifest = gen.read_csv(out_dir / "manifest.csv")
            for row in manifest:
                mask = Image.open(out_dir / row["mask_path"])
                recomputed = coverage_of(mask)
                recorded = float(row["mask_coverage"])
                assert abs(recomputed - recorded) <= 0.5, f"{row['source_id']}/{row['severity']} cov {recomputed:.2f} != {recorded:.2f}"
                if row["restoration_type"] == "local_repair":
                    assert 0.5 <= recomputed <= 35.5, f"{row['source_id']}/{row['severity']} cov {recomputed:.2f} out of [1,35]"
                mask.close()
            return "all coverage in [1,35] and matches metadata"

        # 09 - metadata schema completeness
        def check_metadata() -> str:
            # single metadata/<source_id>.json per source holding every severity variant
            manifest = gen.read_csv(out_dir / "manifest.csv")
            sources_checked = 0
            for row in manifest:
                meta = json.loads((out_dir / row["metadata_path"]).read_text(encoding="utf-8"))
                assert meta.get("source_id") == row["source_id"]
                variants = meta.get("variants", {})
                for sev in SEVERITIES:
                    variant_meta = variants.get(sev)
                    assert variant_meta is not None, f"{row['source_id']} missing variant {sev}"
                    for key in ("seed", "severity", "restoration_type", "mask_coverage", "image_size", "steps"):
                        assert key in variant_meta, f"{row['source_id']}/{sev} missing metadata.{key}"
                    assert variant_meta["severity"] == sev
                    assert variant_meta["steps"], f"{row['source_id']}/{sev} has no steps"
                    assert all("op" in s and "params" in s for s in variant_meta["steps"])
                sources_checked += 1
            return f"{sources_checked} sources: single metadata file with all variants"

        # 10 - global-only mode
        def check_global_only() -> str:
            args = _args(clean_dir=str(clean_dir), output_dir=str(out_global), limit=2, global_only=True)
            sources = gen._select_sources(clean_dir, None, 2)
            gen.generate(args, sources, out_global)
            manifest = gen.read_csv(out_global / "manifest.csv")
            assert len(manifest) == 6
            assert all(r["restoration_type"] == "global_restoration" for r in manifest)
            for row in manifest:
                mask = Image.open(out_global / row["mask_path"])
                assert set(np.asarray(mask).ravel().tolist()) == {0}, f"{row['source_id']}/{row['severity']} expected empty mask"
                mask.close()
            gen.verify(_args(output_dir=str(out_global)), out_global)
            return "global-only: 6 variants, empty masks, verified"

        # 11 - sample mode (pre-batch deliverable for D)
        def check_sample() -> str:
            args = _args(clean_dir=str(clean_dir), output_dir=str(out_sample), sample=2)
            sources = gen._select_sources(clean_dir, None, 2)
            gen.generate(args, sources, out_sample)
            sample_rows = gen.read_csv(out_sample / "sample_manifest.csv")
            assert len(sample_rows) == 6, f"sample manifest has {len(sample_rows)} rows, expected 6"
            assert (out_sample / "contact_sheet_sample.jpg").is_file()
            gen.verify(_args(output_dir=str(out_sample)), out_sample)
            return "sample mode: 6-row sample manifest + sample contact sheet"

        # 12 - idempotent rerun preserves records and bytes
        def check_idempotent() -> str:
            manifest_before = gen.read_csv(out_dir / "manifest.csv")
            degraded_hash = _sha256(out_dir / "degraded" / "fixture_1_light.png")
            args = _args(clean_dir=str(clean_dir), output_dir=str(out_dir), limit=3)
            sources = gen._select_sources(clean_dir, None, 3)
            gen.generate(args, sources, out_dir)
            manifest_after = gen.read_csv(out_dir / "manifest.csv")
            assert len(manifest_after) == len(manifest_before) == 9, "manifest changed on rerun"
            assert _sha256(out_dir / "degraded" / "fixture_1_light.png") == degraded_hash, "bytes changed on rerun"
            return "rerun: manifest preserved, degraded bytes unchanged"

        # 13 - manifest filtering keeps only clean_target_ok, keyed by source_id
        def check_manifest_filter() -> str:
            fake = tmp / "fake_batch"
            accepted = fake / "accepted"
            accepted.mkdir(parents=True)
            for name in ("img_001.jpg", "img_002.png", "img_003.jpg"):
                Image.new("RGB", (64, 64), (128, 128, 128)).save(accepted / name)
            manifest_csv = fake / "manifest.csv"
            gen.write_csv(manifest_csv, [
                {"source_id": "src_001", "filename": "img_001.jpg", "quality_label": "clean_target_ok"},
                {"source_id": "src_002", "filename": "img_002.png", "quality_label": "clean_target_ok"},
                {"source_id": "src_003", "filename": "img_003.jpg", "quality_label": "existing_damage"},
            ], ["source_id", "filename", "quality_label"])
            sources = gen._select_sources(accepted, manifest_csv, None)
            ids = [sid for sid, _ in sources]
            assert ids == ["src_001", "src_002"], f"filtered ids {ids}"
            return "clean_target_ok filtering by source_id"

        # 14 - clear error when neither batch-id nor clean-dir provided
        def check_no_args_error() -> str:
            try:
                gen.main([])
            except SystemExit as exc:
                assert exc.code, "expected a non-zero exit"
                return "SystemExit raised with message"
            raise AssertionError("main([]) did not fail as expected")

        # 15 - empty clean dir exits with code 2
        def check_empty_dir_error() -> str:
            empty = tmp / "empty"
            empty.mkdir()
            try:
                gen.main(["--clean-dir", str(empty), "--output-dir", str(tmp / "empty_out")])
            except SystemExit as exc:
                assert exc.code == 2, f"expected exit 2, got {exc.code}"
                return "empty source dir -> exit 2"
            raise AssertionError("empty dir did not fail")

        # 16 - real CLI invocation works via python -m
        def check_cli_subprocess() -> str:
            env = dict(os.environ)
            src = str(Path(__file__).resolve().parents[2] / "src")
            env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
            sub = str(tmp / "sub_fixture")
            result = subprocess.run(
                [sys.executable, "-m", "scripts.synthetic.generate",
                 "--make-fixture", "--fixture-dir", sub],
                capture_output=True, text=True, env=env,
            )
            assert result.returncode == 0, f"python -m failed: {result.stderr[-500:]}"
            assert len(list(Path(sub).glob("*.png"))) == 3
            return "python -m scripts.synthetic.generate works"

        for name, fn in [
            ("01 engine+CLI import", check_imports),
            ("02 seed derivation", check_seed_derivation),
            ("03 fixture generation", check_fixture),
            ("04 end-to-end generate", check_generate),
            ("05 built-in verify passes", check_verify),
            ("06 determinism (byte-level)", check_determinism),
            ("07 mask properties", check_mask_properties),
            ("08 coverage bounds", check_coverage),
            ("09 metadata schema", check_metadata),
            ("10 global-only mode", check_global_only),
            ("11 sample mode", check_sample),
            ("12 idempotent rerun", check_idempotent),
            ("13 manifest filtering", check_manifest_filter),
            ("14 no-args error", check_no_args_error),
            ("15 empty-dir error", check_empty_dir_error),
            ("16 python -m CLI", check_cli_subprocess),
        ]:
            checks.run(name, fn)

    passed, failed = checks.summary()
    print()
    for name, status, detail in checks._checks:
        print(f"  [{status:4}] {name}: {detail}")
    print()
    print(f"{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
