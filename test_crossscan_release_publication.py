import hashlib
import json
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import crossscan_release_publication as P

TEST_CODE_HEAD = "1" * 40


class CrossscanReleasePublicationTests(unittest.TestCase):
    def _release(self, root: Path) -> Path:
        release = root / "release"
        (release / "model" / "fold_0").mkdir(parents=True)
        (release / "model" / "fold_0" / "checkpoint_final.pth").write_bytes(b"weights\x00\x01")
        (release / "README.md").write_bytes(b"release\n")
        manifest = {
            "schema_version": "crossscan-model-release-v1",
            "status": "PASS",
            "outcome": "POSITIVE_DEPLOYABLE",
            "selected_steps": 4000,
        }
        manifest["content_sha256"] = P.content_hash(manifest)
        (release / P.MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        paths = sorted(
            path.relative_to(release).as_posix()
            for path in release.rglob("*")
            if path.is_file()
        )
        lines = []
        for relative in paths:
            digest = hashlib.sha256((release / relative).read_bytes()).hexdigest()
            lines.append(f"{digest}  {relative}")
        (release / P.CHECKSUM_NAME).write_bytes(("\n".join(lines) + "\n").encode())
        return release

    def test_pack_is_deterministic_and_reverifiable(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            release = self._release(root)
            first_archive = root / "first.tar"
            second_archive = root / "second.tar"
            first_receipt = root / "first.json"
            second_receipt = root / "second.json"
            first = P.pack_release(
                release, first_archive, first_receipt, code_head=TEST_CODE_HEAD
            )
            second = P.pack_release(
                release, second_archive, second_receipt, code_head=TEST_CODE_HEAD
            )
            first_parts = [
                (root / part["path"]).read_bytes() for part in first["archive"]["parts"]
            ]
            second_parts = [
                (root / part["path"]).read_bytes() for part in second["archive"]["parts"]
            ]
            self.assertEqual(first_parts, second_parts)
            self.assertFalse(first_archive.exists())
            self.assertFalse(second_archive.exists())
            self.assertEqual(first["archive"]["sha256"], second["archive"]["sha256"])
            result = P.verify_archive(first_archive, first, release_dir=release)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["members"], 4)

    def test_checksum_writer_finalizes_an_exported_release(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            release = self._release(Path(value))
            (release / P.CHECKSUM_NAME).unlink()
            record = P.write_release_checksums(release)
            self.assertEqual(record["path"], P.CHECKSUM_NAME)
            self.assertEqual(P.validate_release(release)["manifest"]["status"], "PASS")

    def test_archive_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            release = self._release(root)
            archive = root / "release.tar"
            receipt_path = root / "receipt.json"
            receipt = P.pack_release(
                release, archive, receipt_path, code_head=TEST_CODE_HEAD
            )
            first_part = root / receipt["archive"]["parts"][0]["path"]
            payload = bytearray(first_part.read_bytes())
            payload[1024] ^= 1
            first_part.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "archive part checksum mismatch"):
                P.verify_archive(archive, receipt)

    def test_archive_rejects_self_consistent_nonpositive_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            release = self._release(root)
            manifest_path = release / P.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outcome"] = "NEGATIVE_OR_INCONCLUSIVE"
            manifest["content_sha256"] = P.content_hash(manifest)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (release / P.CHECKSUM_NAME).unlink()
            archive = root / "negative.tar"
            with mock.patch.object(
                P, "_validate_positive_manifest", side_effect=lambda value: value
            ):
                P.write_release_checksums(release)
                receipt = P.pack_release(
                    release,
                    archive,
                    root / "negative.json",
                    code_head=TEST_CODE_HEAD,
                )
            with self.assertRaisesRegex(ValueError, "authorized positive"):
                P.verify_archive(archive, receipt)

    def test_release_root_and_archive_part_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            release = self._release(root)
            linked_release = root / "release-link"
            try:
                linked_release.symlink_to(release, target_is_directory=True)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "real directory"):
                P.validate_release(linked_release)

            archive = root / "release.tar"
            receipt = P.pack_release(
                release, archive, root / "receipt.json", code_head=TEST_CODE_HEAD
            )
            part = root / receipt["archive"]["parts"][0]["path"]
            backing = root / "part-backing.bin"
            part.replace(backing)
            part.symlink_to(backing)
            with self.assertRaisesRegex(ValueError, "invalid archive part file"):
                P.verify_archive(archive, receipt)

    def test_crlf_checksum_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            release = self._release(Path(value))
            checksum = release / P.CHECKSUM_NAME
            checksum.write_bytes(checksum.read_bytes().replace(b"\n", b"\r\n"))
            with self.assertRaisesRegex(ValueError, "exactly one LF"):
                P.validate_release(release)

    def test_unlisted_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            release = self._release(Path(value))
            (release / "unlisted.bin").write_bytes(b"not bound")
            with self.assertRaisesRegex(ValueError, "exact release file universe"):
                P.validate_release(release)

    def test_archive_inside_release_is_rejected_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            release = self._release(root)
            archive = release / "bad.tar"
            with self.assertRaisesRegex(ValueError, "outside the release"):
                P.pack_release(
                    release,
                    archive,
                    root / "receipt.json",
                    code_head=TEST_CODE_HEAD,
                )
            self.assertFalse(archive.exists())

    def test_multipart_reader_reconstructs_small_parts(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            release = self._release(root)
            archive = root / "split.tar"
            receipt = P.pack_release(
                release,
                archive,
                root / "split.json",
                code_head=TEST_CODE_HEAD,
                part_size_bytes=1024,
            )
            self.assertGreater(len(receipt["archive"]["parts"]), 1)
            self.assertTrue(all(
                part["bytes"] == 1024
                for part in receipt["archive"]["parts"][:-1]
            ))
            self.assertEqual(P.verify_archive(archive, receipt)["status"], "PASS")

    def test_strict_json_rejects_duplicates_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            P.strict_json_loads('{"value":1,"value":2}')
        with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
            P.strict_json_loads('{"value":NaN}')

    def test_pack_rejects_noncanonical_code_head_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            release = self._release(root)
            archive = root / "release.tar"
            with self.assertRaisesRegex(ValueError, "40 lowercase hexadecimal"):
                P.pack_release(
                    release,
                    archive,
                    root / "receipt.json",
                    code_head="ABC123",
                )
            self.assertFalse(archive.exists())

    def test_publication_validation_binds_payload_to_package_object(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            package_path = root / "package_receipt.json"
            package = P.pack_release(
                self._release(root),
                root / "release.tar",
                package_path,
                code_head=TEST_CODE_HEAD,
            )
            other = dict(package)
            other["code_head"] = "2" * 40
            other["content_sha256"] = P.content_hash(other)
            other_payload = (
                json.dumps(other, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            with self.assertRaisesRegex(ValueError, "payload differs"):
                P.validate_publication_receipt({}, package, other_payload)

    def test_verify_public_checks_dynamic_head_assets_and_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            package_path = root / "package_receipt.json"
            package = P.pack_release(
                self._release(root),
                root / "release.tar",
                package_path,
                code_head=TEST_CODE_HEAD,
                part_size_bytes=1024,
            )
            tag = f"crossscan-v4-{package['release']['manifest']['content_sha256'][:12]}"
            chain = [{
                "type": "commit",
                "sha": TEST_CODE_HEAD,
                "url": f"{P.API_ROOT}/repos/{P.EXPECTED_REPOSITORY}/git/commits/{TEST_CODE_HEAD}",
            }]
            sources = {}
            assets = []
            asset_paths = [package_path] + [
                root / part["path"] for part in package["archive"]["parts"]
            ]
            for index, path in enumerate(asset_paths, start=1):
                digest = P.sha256_file(path)
                url = (
                    f"https://github.com/{P.EXPECTED_REPOSITORY}/releases/"
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
                    "created_at": "2026-08-18T00:00:00Z",
                    "updated_at": "2026-08-18T00:00:00Z",
                    "content_type": "application/octet-stream",
                })
            release_api = {
                "id": 99,
                "html_url": f"https://github.com/{P.EXPECTED_REPOSITORY}/releases/tag/{tag}",
                "tag_name": tag,
                "target_commitish": TEST_CODE_HEAD,
                "draft": False,
                "prerelease": False,
                "immutable": True,
                "published_at": "2026-08-18T00:00:00Z",
                "created_at": "2026-08-18T00:00:00Z",
                "assets": assets,
                "body": "\n".join((
                    TEST_CODE_HEAD,
                    package["archive"]["sha256"],
                    package["release"]["manifest"]["content_sha256"],
                    package["release"]["sha256sums"]["sha256"],
                    package["content_sha256"],
                )),
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
                    "sha256": P.hashlib.sha256(payload).hexdigest(),
                    "headers": {},
                }

            output = root / "publication_receipt.json"
            with self.assertRaisesRegex(ValueError, "independently expected"):
                P.verify_public(
                    package_path,
                    P.EXPECTED_REPOSITORY,
                    tag,
                    output,
                    "2" * 40,
                )
            with (
                mock.patch.object(P, "_resolve_tag", return_value=(TEST_CODE_HEAD, chain)),
                mock.patch.object(P, "_anonymous_json", return_value=release_api),
                mock.patch.object(P, "_download_anonymous", side_effect=fake_download),
            ):
                publication = P.verify_public(
                    package_path,
                    P.EXPECTED_REPOSITORY,
                    tag,
                    output,
                    TEST_CODE_HEAD,
                )
            self.assertEqual(publication["code_head"], TEST_CODE_HEAD)
            self.assertEqual(publication["status"], "PUBLIC_LOGGED_OUT_VERIFIED")
            self.assertEqual(P._load_hashed_json(output), publication)

    def test_anonymous_opener_has_no_proxy_auth_or_cookie_handlers(self) -> None:
        opener = P._anonymous_opener(P.ALLOWED_DOWNLOAD_HOSTS)
        # Supplying ProxyHandler({}) suppresses the default environment-derived
        # handler; because it has no protocols, build_opener does not retain it.
        proxy = [handler for handler in opener.handlers if isinstance(handler, urllib.request.ProxyHandler)]
        self.assertEqual(proxy, [])
        forbidden = (
            urllib.request.AbstractBasicAuthHandler,
            urllib.request.HTTPCookieProcessor,
        )
        self.assertFalse(any(isinstance(handler, forbidden) for handler in opener.handlers))
        redirect = next(handler for handler in opener.handlers if isinstance(handler, P._StrictRedirect))
        request = urllib.request.Request("https://github.com/source")
        with self.assertRaisesRegex(ValueError, "refusing anonymous redirect"):
            redirect.redirect_request(
                request, None, 302, "Found", {}, "http://evil.example/file"
            )
        sensitive = urllib.request.Request(
            "https://github.com/source",
            headers={"Authorization": "secret", "Cookie": "session"},
        )
        sensitive.add_unredirected_header("Proxy-Authorization", "proxy-secret")
        redirected = redirect.redirect_request(
            sensitive, None, 302, "Found", {}, "https://github.com/target"
        )
        names = {name.lower() for name, _ in redirected.header_items()}
        self.assertTrue({"authorization", "cookie", "proxy-authorization"}.isdisjoint(names))


if __name__ == "__main__":
    unittest.main()
