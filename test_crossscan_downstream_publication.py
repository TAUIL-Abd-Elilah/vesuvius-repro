from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import crossscan_downstream_publication as C
import crossscan_release_publication as M
import crossscan_scrollfiesta_adapter as A
import run_crossscan_scrollfiesta_downstream as D


DOWNSTREAM_CODE_HEAD = "2" * 40
MODEL_CODE_HEAD = "1" * 40


class CrossscanDownstreamPublicationTests(unittest.TestCase):
    def _model_publication(self, root: Path) -> tuple[Path, Path, dict]:
        release = root / "model_release"
        (release / "model" / "fold_0").mkdir(parents=True)
        (release / "model" / "fold_0" / "checkpoint_final.pth").write_bytes(b"weights\x00\x01")
        manifest = {
            "schema_version": "crossscan-model-release-v1",
            "status": "PASS",
            "outcome": "POSITIVE_DEPLOYABLE",
            "selected_steps": 4000,
        }
        manifest["content_sha256"] = M.content_hash(manifest)
        (release / M.MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        paths = sorted(
            path.relative_to(release).as_posix()
            for path in release.rglob("*")
            if path.is_file()
        )
        checksum_lines = [
            f"{hashlib.sha256((release / relative).read_bytes()).hexdigest()}  {relative}"
            for relative in paths
        ]
        (release / M.CHECKSUM_NAME).write_bytes(("\n".join(checksum_lines) + "\n").encode())
        publication_root = root / "model_publication"
        publication_root.mkdir()
        package_path = publication_root / "package_receipt.json"
        package = M.pack_release(
            release,
            publication_root / "model.tar",
            package_path,
            code_head=MODEL_CODE_HEAD,
            part_size_bytes=1024,
        )
        tag = f"crossscan-v4-{manifest['content_sha256'][:12]}"
        publication = {
            "schema_version": M.PUBLICATION_SCHEMA,
            "status": "PUBLIC_LOGGED_OUT_VERIFIED",
            "repository": M.EXPECTED_REPOSITORY,
            "code_head": MODEL_CODE_HEAD,
            "tag": tag,
            "release_manifest_content_sha256": manifest["content_sha256"],
            "package_receipt": {"content_sha256": package["content_sha256"]},
            "verifier": dict(C.UPSTREAM_ALLOWED_HISTORICAL_VERIFIERS[0]),
        }
        publication["content_sha256"] = M.content_hash(publication)
        publication_path = publication_root / "publication_receipt.json"
        publication_path.write_text(
            json.dumps(publication, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return package_path, publication_path, package

    @contextlib.contextmanager
    def _trusted_model(self):
        with (
            mock.patch.object(M, "validate_publication_receipt", return_value={}),
            mock.patch.object(M, "probe_publication", return_value={"status": "PASS"}),
        ):
            yield

    def _downstream(
        self,
        root: Path,
        model_package: dict,
        *,
        scientific_status: str = "FAIL",
    ) -> Path:
        downstream = root / f"downstream_{scientific_status.lower()}"
        downstream.mkdir()
        model_manifest = root / "model_release" / M.MANIFEST_NAME
        for arm in ("candidate-fixed", "candidate-matched-mass"):
            destination = (
                downstream / "inputs" / "grids" / arm / "provenance" / "release_manifest.json"
            )
            destination.parent.mkdir(parents=True)
            shutil.copyfile(model_manifest, destination)
        provenance = downstream / "provenance"
        provenance.mkdir()
        for name in (
            "crossscan_scrollfiesta_downstream_lock.json",
            "crossscan_scrollfiesta_metric_lock.json",
        ):
            shutil.copyfile(Path(D.__file__).with_name(name), provenance / name)
        (downstream / "empty-preserved-directory").mkdir()
        (downstream / "artifact.bin").write_bytes(b"sealed downstream payload")
        passed = scientific_status == "PASS"
        receipt = {
            "schema_version": D.SCHEMA,
            "status": scientific_status,
            "downstream_lock_content_sha256": A.DOWNSTREAM_LOCK_CONTENT_SHA256,
            "metric_lock_content_sha256": D.METRIC_LOCK_CONTENT_SHA256,
            "physical": {"pass": passed},
            "scrollfiesta_gate": {"pass": passed},
            "visual_evidence": {"pass": passed},
            "input_integrity": {"pass": passed},
            "artifact_integrity": {"invalid_output_entries": [], "pass": True},
            "terminal_gate": {
                "physical_pass": passed,
                "scrollfiesta_pass": passed,
                "visuals_pass": passed,
                "input_integrity_pass": passed,
                "artifact_integrity_pass": True,
                "pass": passed,
            },
            "claim_boundary": (
                "bounded untouched-PHerc0139 probability-to-ScrollFiesta improvement"
                if passed
                else "bounded negative result; no downstream-improvement claim is authorized"
            ),
            "files": D._hash_tree(downstream),
        }
        receipt["content_sha256"] = A.content_hash(receipt)
        D._write_json_exclusive(downstream / C.TERMINAL_NAME, receipt)
        return downstream

    def _pack(
        self,
        root: Path,
        downstream: Path,
        model_package_path: Path,
        model_publication_path: Path,
        *,
        name: str = "downstream",
        part_size_bytes: int = 1024,
    ) -> tuple[dict, Path, Path]:
        archive = root / f"{name}.tar"
        receipt = root / f"{name}.json"
        terminal_digest = json.loads(
            (downstream / C.TERMINAL_NAME).read_text(encoding="utf-8")
        )["content_sha256"]
        upstream_digest = json.loads(
            model_publication_path.read_text(encoding="utf-8")
        )["content_sha256"]
        with self._trusted_model():
            with mock.patch.object(D, "verify_result", side_effect=self._fixture_verify_result):
                package = C.pack_downstream(
                    downstream,
                    model_package_path,
                    model_publication_path,
                    archive,
                    receipt,
                    code_head=DOWNSTREAM_CODE_HEAD,
                    expected_terminal_content_sha256=terminal_digest,
                    expected_upstream_publication_content_sha256=upstream_digest,
                    part_size_bytes=part_size_bytes,
                    probe_upstream=False,
                )
        return package, archive, receipt

    @staticmethod
    def _fixture_verify_result(path: Path, **_kwargs):
        return json.loads(
            (Path(path) / C.TERMINAL_NAME).read_text(encoding="utf-8")
        )

    def test_fail_tree_packages_deterministically_and_ignores_unsealed_empty_directories(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            model_path, publication_path, model = self._model_publication(root)
            downstream = self._downstream(root, model)
            first, first_archive, _ = self._pack(
                root, downstream, model_path, publication_path, name="first"
            )
            second, second_archive, _ = self._pack(
                root, downstream, model_path, publication_path, name="second"
            )
            first_bytes = b"".join(
                (root / part["path"]).read_bytes() for part in first["archive"]["parts"]
            )
            second_bytes = b"".join(
                (root / part["path"]).read_bytes() for part in second["archive"]["parts"]
            )
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first["archive"]["sha256"], second["archive"]["sha256"])
            self.assertFalse(first_archive.exists())
            self.assertFalse(second_archive.exists())
            self.assertNotIn("directories", first["downstream"])
            self.assertEqual(first["downstream"]["scientific_status"], "FAIL")
            self.assertEqual(first["status"], "PASS")
            with self._trusted_model(), mock.patch.object(
                D, "verify_result", side_effect=self._fixture_verify_result
            ):
                verified = C.verify_archive(
                    first_archive,
                    first,
                    model_package_receipt=model_path,
                    model_publication_receipt=publication_path,
                    deep=True,
                    probe_upstream=False,
                    expected_terminal_content_sha256=(
                        first["downstream"]["terminal_receipt"]["content_sha256"]
                    ),
                    expected_upstream_publication_content_sha256=(
                        first["upstream_model_publication"]["publication_receipt"][
                            "content_sha256"
                        ]
                    ),
                )
            self.assertEqual(verified["scientific_status"], "FAIL")

    def test_dynamic_confirmation_binds_status_and_full_terminal_digest(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            model_path, publication_path, model = self._model_publication(root)
            downstream = self._downstream(root, model)
            package, _, _ = self._pack(root, downstream, model_path, publication_path)
            digest = package["downstream"]["terminal_receipt"]["content_sha256"]
            self.assertEqual(
                C.expected_confirmation(package),
                (
                    f"PUBLISH IMMUTABLE CROSSSCAN V4 DOWNSTREAM FAIL {digest} "
                    f"AT {DOWNSTREAM_CODE_HEAD}"
                ),
            )
            self.assertEqual(C.expected_tag(package), f"crossscan-v4-downstream-{digest[:12]}")

    def test_verifier_source_identity_is_crlf_lf_portable_but_historical_tuple_is_exact(self):
        source = Path(M.__file__).read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(
            C._source_identity(source, "crossscan_release_publication.py"),
            C.UPSTREAM_PUBLICATION_SOURCE_IDENTITY,
        )
        self.assertEqual(
            C._source_identity(
                source.replace(b"\n", b"\r\n"),
                "crossscan_release_publication.py",
            ),
            C.UPSTREAM_PUBLICATION_SOURCE_IDENTITY,
        )
        downstream_source = Path(C.__file__).read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(
            C._source_identity(
                downstream_source, "crossscan_downstream_publication.py"
            ),
            C._source_identity(
                downstream_source.replace(b"\n", b"\r\n"),
                "crossscan_downstream_publication.py",
            ),
        )

        with tempfile.TemporaryDirectory() as value:
            package_path, publication_path, _ = self._model_publication(Path(value))
            package, payload = C._load_hashed_json(package_path, M.PACKAGE_SCHEMA)
            publication, _ = C._load_hashed_json(publication_path, M.PUBLICATION_SCHEMA)
            with mock.patch.object(M, "validate_publication_receipt", return_value={}):
                C._upstream_validation_surrogate(publication, package, payload)
            publication["verifier"] = {
                **publication["verifier"], "script_sha256": "0" * 64,
            }
            publication["content_sha256"] = M.content_hash(publication)
            with self.assertRaisesRegex(ValueError, "unapproved verifier/runtime"):
                C._upstream_validation_surrogate(publication, package, payload)

    def test_canonical_release_body_has_exact_claim_boundary_and_final_lf(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            model_path, publication_path, model = self._model_publication(root)
            downstream = self._downstream(root, model)
            package, _, _ = self._pack(root, downstream, model_path, publication_path)
            with self._trusted_model():
                body = C.canonical_release_body(package)
            self.assertTrue(body.endswith("\n"))
            self.assertFalse(body.endswith("\n\n"))
            self.assertIn(
                "\nBounded negative result; no downstream-improvement claim is authorized.\n",
                body,
            )
            self.assertIn(
                f"- Downstream tooling code head: {DOWNSTREAM_CODE_HEAD}\n", body
            )
            self.assertNotEqual(body, body + "\n")

    def test_independent_terminal_and_upstream_pins_are_required_before_outputs(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            model_path, publication_path, model = self._model_publication(root)
            downstream = self._downstream(root, model)
            upstream_digest = json.loads(publication_path.read_text(encoding="utf-8"))[
                "content_sha256"
            ]
            with (
                self._trusted_model(),
                mock.patch.object(D, "verify_result", side_effect=self._fixture_verify_result),
                self.assertRaisesRegex(ValueError, "independently expected terminal"),
            ):
                C.pack_downstream(
                    downstream,
                    model_path,
                    publication_path,
                    root / "wrong-terminal.tar",
                    root / "wrong-terminal.json",
                    code_head=DOWNSTREAM_CODE_HEAD,
                    expected_terminal_content_sha256="a" * 64,
                    expected_upstream_publication_content_sha256=upstream_digest,
                    probe_upstream=False,
                )
            self.assertFalse((root / "wrong-terminal.tar.tmp").exists())
            self.assertFalse(list(root.glob("wrong-terminal.tar.part-*")))
            terminal_digest = json.loads(
                (downstream / C.TERMINAL_NAME).read_text(encoding="utf-8")
            )["content_sha256"]
            with (
                self._trusted_model(),
                mock.patch.object(D, "verify_result", side_effect=self._fixture_verify_result),
                self.assertRaisesRegex(ValueError, "independently expected publication"),
            ):
                C.pack_downstream(
                    downstream,
                    model_path,
                    publication_path,
                    root / "wrong-upstream.tar",
                    root / "wrong-upstream.json",
                    code_head=DOWNSTREAM_CODE_HEAD,
                    expected_terminal_content_sha256=terminal_digest,
                    expected_upstream_publication_content_sha256="b" * 64,
                    probe_upstream=False,
                )
            self.assertFalse((root / "wrong-upstream.tar.tmp").exists())

    def test_candidate_grid_must_embed_exact_published_model_manifest(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            model_path, publication_path, model = self._model_publication(root)
            downstream = self._downstream(root, model)
            embedded = (
                downstream / "inputs" / "grids" / "candidate-fixed" /
                "provenance" / "release_manifest.json"
            )
            value_json = json.loads(embedded.read_text(encoding="utf-8"))
            value_json["selected_steps"] = 3999
            value_json["content_sha256"] = M.content_hash(value_json)
            embedded.write_text(
                json.dumps(value_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            receipt_path = downstream / C.TERMINAL_NAME
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["files"] = D._hash_tree(downstream, excluded=(C.TERMINAL_NAME,))
            receipt["content_sha256"] = A.content_hash(receipt)
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            with (
                self._trusted_model(),
                mock.patch.object(D, "verify_result", side_effect=self._fixture_verify_result),
                self.assertRaisesRegex(ValueError, "different model release"),
            ):
                C.pack_downstream(
                    downstream,
                    model_path,
                    publication_path,
                    root / "bad.tar",
                    root / "bad.json",
                    code_head=DOWNSTREAM_CODE_HEAD,
                    expected_terminal_content_sha256=receipt["content_sha256"],
                    expected_upstream_publication_content_sha256=json.loads(
                        publication_path.read_text(encoding="utf-8")
                    )["content_sha256"],
                    probe_upstream=False,
                )

    def test_archive_part_tamper_and_terminal_link_are_rejected(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            model_path, publication_path, model = self._model_publication(root)
            downstream = self._downstream(root, model)
            package, archive, _ = self._pack(root, downstream, model_path, publication_path)
            part = root / package["archive"]["parts"][0]["path"]
            payload = bytearray(part.read_bytes())
            payload[0] ^= 1
            part.write_bytes(payload)
            with self._trusted_model(), self.assertRaisesRegex(
                ValueError, "archive part checksum mismatch"
            ):
                C.verify_archive(
                    archive,
                    package,
                    deep=False,
                    probe_upstream=False,
                    expected_terminal_content_sha256=(
                        package["downstream"]["terminal_receipt"]["content_sha256"]
                    ),
                    expected_upstream_publication_content_sha256=(
                        package["upstream_model_publication"]["publication_receipt"][
                            "content_sha256"
                        ]
                    ),
                )

            terminal = downstream / C.TERMINAL_NAME
            backing = root / "terminal-backing.json"
            terminal.replace(backing)
            try:
                terminal.symlink_to(backing)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "linked or special downstream file"):
                C.validate_downstream_tree(downstream)

    def test_embedded_upstream_receipt_payload_and_part_universe_are_strict(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            model_path, publication_path, model = self._model_publication(root)
            downstream = self._downstream(root, model)
            package, archive, _ = self._pack(root, downstream, model_path, publication_path)
            forged = json.loads(json.dumps(package))
            forged["upstream_model_publication"]["publication_receipt"]["payload_base64"] = "AAAA"
            forged["content_sha256"] = C.content_hash(forged)
            with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                C._validate_package_receipt(forged)

            forged = json.loads(json.dumps(package))
            forged["upstream_model_publication"]["code_head"] = "9" * 40
            forged["content_sha256"] = C.content_hash(forged)
            with self._trusted_model(), self.assertRaisesRegex(
                ValueError, "summary differs"
            ):
                C._validate_package_receipt(forged)

            extra = root / f"{archive.name}.part-999"
            extra.write_bytes(b"extra")
            with self._trusted_model(), self.assertRaisesRegex(ValueError, "file universe differs"):
                C.verify_archive(
                    archive,
                    package,
                    deep=False,
                    probe_upstream=False,
                    expected_terminal_content_sha256=(
                        package["downstream"]["terminal_receipt"]["content_sha256"]
                    ),
                    expected_upstream_publication_content_sha256=(
                        package["upstream_model_publication"]["publication_receipt"][
                            "content_sha256"
                        ]
                    ),
                )

    def test_strict_terminal_json_and_part_ceiling_are_enforced(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            model_path, publication_path, model = self._model_publication(root)
            downstream = self._downstream(root, model)
            terminal = downstream / C.TERMINAL_NAME
            payload = terminal.read_text(encoding="utf-8")
            terminal.write_text(
                payload.replace('"status": "FAIL",', '"status": "FAIL",\n  "status": "FAIL",', 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                C.validate_downstream_tree(downstream)

            terminal.write_text(payload, encoding="utf-8")
            terminal_digest = json.loads(
                (downstream / C.TERMINAL_NAME).read_text(encoding="utf-8")
            )["content_sha256"]
            upstream_digest = json.loads(publication_path.read_text(encoding="utf-8"))[
                "content_sha256"
            ]
            with (
                self._trusted_model(),
                mock.patch.object(D, "verify_result", side_effect=self._fixture_verify_result),
                mock.patch.object(C, "MAX_ARCHIVE_PARTS", 1),
                self.assertRaisesRegex(ValueError, "maximum is 1"),
            ):
                C.pack_downstream(
                    downstream,
                    model_path,
                    publication_path,
                    root / "too-many.tar",
                    root / "too-many.json",
                    code_head=DOWNSTREAM_CODE_HEAD,
                    expected_terminal_content_sha256=terminal_digest,
                    expected_upstream_publication_content_sha256=upstream_digest,
                    part_size_bytes=1024,
                    probe_upstream=False,
                )

    def test_pass_and_fail_are_both_packageable_but_forged_pass_needs_deep_verifier(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            model_path, publication_path, model = self._model_publication(root)
            downstream = self._downstream(root, model, scientific_status="PASS")

            def fixture_verify(path: Path, **_kwargs):
                return json.loads((Path(path) / C.TERMINAL_NAME).read_text(encoding="utf-8"))

            with self._trusted_model(), self.assertRaises((FileNotFoundError, ValueError)):
                C.pack_downstream(
                    downstream,
                    model_path,
                    publication_path,
                    root / "forged-pass.tar",
                    root / "forged-pass.json",
                    code_head=DOWNSTREAM_CODE_HEAD,
                    expected_terminal_content_sha256=json.loads(
                        (downstream / C.TERMINAL_NAME).read_text(encoding="utf-8")
                    )["content_sha256"],
                    expected_upstream_publication_content_sha256=json.loads(
                        publication_path.read_text(encoding="utf-8")
                    )["content_sha256"],
                    probe_upstream=False,
                )
            with self._trusted_model(), mock.patch.object(D, "verify_result", side_effect=fixture_verify):
                package = C.pack_downstream(
                    downstream,
                    model_path,
                    publication_path,
                    root / "pass.tar",
                    root / "pass.json",
                    code_head=DOWNSTREAM_CODE_HEAD,
                    expected_terminal_content_sha256=json.loads(
                        (downstream / C.TERMINAL_NAME).read_text(encoding="utf-8")
                    )["content_sha256"],
                    expected_upstream_publication_content_sha256=json.loads(
                        publication_path.read_text(encoding="utf-8")
                    )["content_sha256"],
                    part_size_bytes=1024,
                    probe_upstream=False,
                )
            self.assertEqual(package["downstream"]["scientific_status"], "PASS")
            self.assertTrue(package["downstream"]["verification"]["pass_recomputed"])
            with self.assertRaises((FileNotFoundError, ValueError)):
                D.verify_result(downstream, deep=True)

    def test_public_readback_binds_tag_head_assets_upstream_and_scientific_status(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            model_path, model_publication_path, model = self._model_publication(root)
            downstream = self._downstream(root, model)
            package, _, receipt = self._pack(
                root,
                downstream,
                model_path,
                model_publication_path,
                name="public-downstream",
            )
            fixed_receipt = root / C.PACKAGE_ASSET_NAME
            receipt.replace(fixed_receipt)
            tag = C.expected_tag(package)
            chain = [{
                "type": "commit",
                "sha": DOWNSTREAM_CODE_HEAD,
                "url": f"{M.API_ROOT}/repos/{C.EXPECTED_REPOSITORY}/git/commits/{DOWNSTREAM_CODE_HEAD}",
            }]
            sources: dict[str, Path] = {}
            assets = []
            asset_paths = [fixed_receipt] + [
                root / part["path"] for part in package["archive"]["parts"]
            ]
            for index, path in enumerate(asset_paths, start=1):
                digest = C.sha256_file(path)
                url = (
                    f"https://github.com/{C.EXPECTED_REPOSITORY}/releases/"
                    f"download/{tag}/{path.name}"
                )
                sources[url] = path
                assets.append({
                    "id": index,
                    "node_id": f"asset-{index}",
                    "name": path.name,
                    "state": "uploaded",
                    "size": path.stat().st_size,
                    "digest": f"sha256:{digest}",
                    "browser_download_url": url,
                    "created_at": "2026-08-19T00:00:00Z",
                    "updated_at": "2026-08-19T00:00:00Z",
                    "content_type": "application/octet-stream",
                })
            with self._trusted_model():
                release_body = C.canonical_release_body(package)
            release_api = {
                "id": 123,
                "html_url": f"https://github.com/{C.EXPECTED_REPOSITORY}/releases/tag/{tag}",
                "tag_name": tag,
                "target_commitish": DOWNSTREAM_CODE_HEAD,
                "draft": False,
                "prerelease": False,
                "immutable": True,
                "published_at": "2026-08-19T00:00:00Z",
                "created_at": "2026-08-19T00:00:00Z",
                "assets": assets,
                "body": release_body,
            }

            def fake_download(url: str, output: Path, expected_bytes: int):
                payload = sources[url].read_bytes()
                self.assertEqual(len(payload), expected_bytes)
                output.write_bytes(payload)
                return {
                    "authorization_header_sent": False,
                    "cookies_enabled": False,
                    "environment_proxy_enabled": False,
                    "accept_encoding": "identity",
                    "resolved_host": "github.com",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "headers": {},
                }

            output = root / C.PUBLICATION_RECEIPT_NAME
            terminal_digest = package["downstream"]["terminal_receipt"]["content_sha256"]
            upstream_digest = package["upstream_model_publication"]["publication_receipt"][
                "content_sha256"
            ]
            with self._trusted_model(), self.assertRaisesRegex(ValueError, "independently expected"):
                C.verify_public(
                    fixed_receipt,
                    None,
                    None,
                    C.EXPECTED_REPOSITORY,
                    tag,
                    output,
                    "3" * 40,
                    terminal_digest,
                    upstream_digest,
                )
            release_api["body"] += "\n"
            with (
                self._trusted_model(),
                mock.patch.object(C, "_resolve_tag", return_value=(DOWNSTREAM_CODE_HEAD, chain)),
                mock.patch.object(C, "_anonymous_json", return_value=release_api),
                self.assertRaisesRegex(ValueError, "authorized exact body"),
            ):
                C.verify_public(
                    fixed_receipt,
                    None,
                    None,
                    C.EXPECTED_REPOSITORY,
                    tag,
                    output,
                    DOWNSTREAM_CODE_HEAD,
                    terminal_digest,
                    upstream_digest,
                )
            release_api["body"] = release_body
            with (
                self._trusted_model(),
                mock.patch.object(C, "_resolve_tag", return_value=(DOWNSTREAM_CODE_HEAD, chain)),
                mock.patch.object(C, "_anonymous_json", return_value=release_api),
                mock.patch.object(C, "_download_anonymous", side_effect=fake_download),
                mock.patch.object(D, "verify_result", side_effect=self._fixture_verify_result),
            ):
                publication = C.verify_public(
                    fixed_receipt,
                    None,
                    None,
                    C.EXPECTED_REPOSITORY,
                    tag,
                    output,
                    DOWNSTREAM_CODE_HEAD,
                    terminal_digest,
                    upstream_digest,
                )
            self.assertEqual(publication["status"], "PUBLIC_LOGGED_OUT_VERIFIED")
            self.assertEqual(publication["scientific_status"], "FAIL")
            self.assertEqual(publication["terminal_content_sha256"], package["downstream"]["terminal_receipt"]["content_sha256"])
            loaded, _ = C._load_hashed_json(output, C.PUBLICATION_SCHEMA)
            self.assertEqual(loaded, publication)

    def test_publisher_uses_resumable_draft_release_id_contract(self):
        source = Path(C.__file__).with_name(
            "publish_crossscan_v4_downstream.ps1"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--jq", source)
        self.assertNotIn("releases/tags/", source)
        self.assertNotIn("gh release edit", source)
        self.assertIn("gh api --paginate --slurp", source)
        self.assertIn("Where-Object { [string]$_.tag_name -ceq $tag }", source)
        self.assertIn("https://uploads.github.com/repos/$repository/releases/$releaseId/assets", source)
        self.assertIn('if ($seenAssets.ContainsKey([string]$upload.name))', source)
        self.assertIn('-F draft=false', source)
        self.assertIn("function Get-CanonicalReleaseBody", source)
        self.assertIn("$scientificStatus $terminalDigest AT $ExpectedCodeHead", source)
        self.assertIn("Assert-ReleaseStaticMetadata", source)
        self.assertIn("Assert-ExactAssetUniverse", source)
        self.assertIn("Get-GitHubReleaseById", source)
        self.assertIn("gh api --method PATCH", source)


if __name__ == "__main__":
    unittest.main()
