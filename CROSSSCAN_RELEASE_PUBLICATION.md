# Cross-scan release publication

`crossscan_release_publication.py` packages an already verified positive-only
cross-scan release. It never uploads or changes GitHub state. The exporter
creates the required LF-only `SHA256SUMS`; `pack` independently checks that it
binds every release file other than itself.

## Deterministic package

```powershell
python crossscan_release_publication.py pack `
  --release-dir D:\release\crossscan_release_v4 `
  --archive D:\publication\crossscan_release_v4.tar `
  --receipt D:\publication\package_receipt.json `
  --code-head <40-character-public-commit>
```

The logical output is an uncompressed deterministic PAX tar under the single
root `crossscan_release_v4/`. To remain below GitHub's per-asset limit, it is
stored as ordered `crossscan_release_v4.tar.part-NNN` files of at most
1,900,000,000 bytes. No unsplit tar remains after verification.

`--code-head` is recorded in the self-hashed package receipt. Public
verification requires both the release tag and its resolved tag-object chain
to target that exact commit.

The command rejects links, reparse points, special files, non-portable paths,
CRLF/BOM checksum files, duplicate/case-alias paths, an incomplete checksum
universe, invalid canonical manifest hashes, and any non-positive release. It
fixes every tar member's time, ownership, mode, type, and order; binds each part
and the reconstructed tar in `package_receipt.json`; then reopens the package
through the multipart reader and verifies every archived file.

Reverify a local package without reconstructing a second full tar:

```powershell
python crossscan_release_publication.py verify `
  --archive D:\publication\crossscan_release_v4.tar `
  --receipt D:\publication\package_receipt.json `
  --release-dir D:\release\crossscan_release_v4
```

The `--archive` path is the logical basename; the verifier consumes the ordered
part files next to it.

## Anonymous immutable-release verification

Upload exactly the part files plus `package_receipt.json` to a final GitHub
release whose tag targets the pinned public code commit. Enable GitHub immutable
releases before publishing the draft. Release notes must contain the code head,
logical archive SHA-256, release-manifest content SHA-256, SHA256SUMS file
SHA-256, and package-receipt content SHA-256.

```powershell
python crossscan_release_publication.py verify-public `
  --package-receipt D:\publication\package_receipt.json `
  --repository TAUIL-Abd-Elilah/vesuvius-repro `
  --tag crossscan-v4-<first-12-manifest-content-hash> `
  --output D:\publication\publication_receipt.json `
  --expected-code-head <40-character-public-commit>
```

The independently supplied expected head must equal the self-hashed package
receipt and the resolved public tag target. The verifier then uses a dedicated
no-proxy/no-auth/no-cookie opener, permits only
strict HTTPS redirects to GitHub release hosts, checks the tag chain and exact
asset universe/server-side digests, downloads every asset without credentials,
revalidates the reconstructed archive, and records a self-hashed publication
receipt. Downstream execution should call `validate_publication_receipt` and
`probe_publication` before accepting that receipt.

## Tests

```powershell
python -m unittest -v test_crossscan_release_publication.py
```

The tests cover deterministic multipart reconstruction, exact-universe and
line-ending failures, link and part tampering, output placement, strict JSON
parsing, public release metadata/download verification, and the anonymous
network-handler policy. The test file is included in the exported release.
