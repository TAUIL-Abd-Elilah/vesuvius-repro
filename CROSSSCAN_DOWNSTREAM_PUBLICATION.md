# Cross-scan downstream publication

This is the public judge/operator runbook for packaging and publishing an
already sealed Cross-scan v4 ScrollFiesta downstream result. The publication
tools do not run the experiment or change its verdict.

## Scientific PASS is not package PASS

The terminal receipt contains the scientific verdict:

- `PASS` authorizes only the preregistered bounded untouched-PHerc0139
  probability-to-ScrollFiesta improvement claim.
- `FAIL` is the sealed negative result and authorizes no downstream-improvement
  claim.

By contrast, `downstream_package_receipt.json.status == "PASS"` and a `verify`
result with `status == "PASS"` mean that package integrity passed. Read the
adjacent `scientific_status` to learn the scientific result. A FAIL result can
and should have a package PASS. Likewise,
`downstream_publication_receipt.json.status == "PUBLIC_LOGGED_OUT_VERIFIED"`
means that the sealed result was read back successfully; it does not turn a
scientific FAIL into a PASS.

Publish either terminal verdict. Do not rerun, replace, or selectively omit a
sealed FAIL in search of a PASS.

## Exact identity chain

Record the terminal content SHA-256 and upstream publication-receipt content
SHA-256 independently before packaging. Do not copy either expected value from
the package that it is meant to check.

The package binds all of the following:

- the terminal receipt, its PASS/FAIL verdict, and the exact regular-file
  universe named by that receipt;
- downstream lock
  `06142dc819c193a462f37d08a4769024c41ab551411013d40dda72db148457f6`
  and metric lock
  `70c29b370b1f6ca2bb7f6d78eb284e456187056d2ed7efb86c7b5950e976f42c`;
- the public code commit supplied as `--code-head`;
- upstream files named exactly `package_receipt.json` and
  `publication_receipt.json`, including their bytes and self-hashes;
- the upstream `TAUIL-Abd-Elilah/vesuvius-repro` tag
  `crossscan-v4-<first-12-release-manifest-content-hash>`, its code head,
  release manifest, package receipt, and archive size/hash; and
- both candidate grid copies of `release_manifest.json`, byte-for-byte and by
  canonical content hash, to that same upstream model publication.

The upstream publication receipt must already say
`PUBLIC_LOGGED_OUT_VERIFIED`. Packaging validates it with the upstream release
publication verifier and performs a live anonymous probe. The exact upstream
receipts are embedded in the downstream package receipt so a later anonymous
judge can recheck the full chain.

## Pack and verify locally

Use fresh output paths outside the sealed downstream directory. Outputs are
exclusive: the packer will not overwrite a receipt, archive part, or staging
file.

```powershell
$terminal = "<64-lowercase-hex terminal content SHA-256>"
$upstreamPublication = "<64-lowercase-hex upstream publication-receipt content SHA-256>"
$head = "<40-lowercase-hex public code commit>"

python crossscan_downstream_publication.py pack `
  --downstream-dir D:\data\crossscan_scrollfiesta_v4\downstream `
  --model-package-receipt D:\data\crossscan_release_v4_publication\package_receipt.json `
  --model-publication-receipt D:\data\crossscan_release_v4_publication\publication_receipt.json `
  --archive D:\publication\crossscan_scrollfiesta_v4_downstream.tar `
  --receipt D:\publication\downstream_package_receipt.json `
  --code-head $head `
  --expected-terminal-content-sha256 $terminal `
  --expected-upstream-publication-content-sha256 $upstreamPublication
```

The logical archive is a deterministic uncompressed PAX tar, stored as ordered
`.part-NNN` files next to the logical `.tar` path. The unsplit staging tar is
removed after two verification passes. Reverify the parts and the original
sealed tree with:

```powershell
python crossscan_downstream_publication.py verify `
  --archive D:\publication\crossscan_scrollfiesta_v4_downstream.tar `
  --receipt D:\publication\downstream_package_receipt.json `
  --downstream-dir D:\data\crossscan_scrollfiesta_v4\downstream `
  --model-package-receipt D:\data\crossscan_release_v4_publication\package_receipt.json `
  --model-publication-receipt D:\data\crossscan_release_v4_publication\publication_receipt.json `
  --expected-terminal-content-sha256 $terminal `
  --expected-upstream-publication-content-sha256 $upstreamPublication
```

`--archive` remains the logical tar name even though verification reads its
ordered parts. Check both `status` and `scientific_status` in the JSON output.

## Publish the immutable release

The tag is derived, not chosen:

```text
crossscan-v4-downstream-<first 12 characters of terminal content SHA-256>
```

It must resolve to the package's exact code head. Generate the case-sensitive
confirmation from the validated receipt rather than typing or hard-coding it:

```powershell
$python = (Get-Command python).Source
$confirmation = (& $python crossscan_downstream_publication.py confirmation `
  --receipt D:\publication\downstream_package_receipt.json).Trim()
$confirmation
```

Its exact form is dynamic:

```text
PUBLISH IMMUTABLE CROSSSCAN V4 DOWNSTREAM <PASS|FAIL> <full terminal content SHA-256> AT <40-character code head>
```

From a clean worktree at the public package code head, with `gh` authenticated
as `TAUIL-Abd-Elilah`, run:

```powershell
.\publish_crossscan_v4_downstream.ps1 `
  -Confirmation $confirmation `
  -ExpectedCodeHead $head `
  -ExpectedTerminalContentSHA256 $terminal `
  -ExpectedUpstreamPublicationContentSHA256 $upstreamPublication `
  -Python $python `
  -DownstreamDir D:\data\crossscan_scrollfiesta_v4\downstream `
  -PublicationRoot D:\publication `
  -ModelPackageReceipt D:\data\crossscan_release_v4_publication\package_receipt.json `
  -ModelPublicationReceipt D:\data\crossscan_release_v4_publication\publication_receipt.json
```

The wrapper requires origin
`https://github.com/TAUIL-Abd-Elilah/vesuvius-repro.git`, an independently
supplied code head at the approved
`refs/heads/physical-crossscan-release-tooling` ref, repository admin
permission, and GitHub immutable releases. It
enables and rechecks the immutable-release setting if necessary. It then
creates a draft, uploads exactly the ordered parts plus
`downstream_package_receipt.json`, verifies release notes and server-side
digests, and only then makes that same release public and immutable.

### Safe draft resume

After an interrupted draft upload, preserve the draft and rerun the exact same
command. Resume is allowed only for one draft with the exact tag, target, title,
notes, and expected asset subset. Already uploaded assets must match size and
server SHA-256. An incomplete `starter` asset aborts safely and is never
deleted automatically; missing asset names are uploaded only when no starter
or mismatch exists. Every unexpected, duplicate, or mismatched asset aborts
publication.

Do not manually replace assets or create a second tag. If the release already
became public and immutable but receipt creation was interrupted, rerun the
exact same publisher command. It recognizes the exact immutable release and
enters readback-only recovery; it never uploads or patches that release again.

## Anonymous public readback

The publisher runs this step automatically after publication. A judge can also
run it independently without GitHub credentials:

```powershell
$tag = "crossscan-v4-downstream-$($terminal.Substring(0, 12))"

python crossscan_downstream_publication.py verify-public `
  --package-receipt D:\publication\downstream_package_receipt.json `
  --repository TAUIL-Abd-Elilah/vesuvius-repro `
  --tag $tag `
  --output D:\publication\downstream_publication_receipt.json `
  --expected-code-head $head `
  --expected-terminal-content-sha256 $terminal `
  --expected-upstream-publication-content-sha256 $upstreamPublication
```

The verifier uses a dedicated no-proxy, no-auth, no-cookie HTTPS client. It
requires a final immutable release, resolves the tag chain to the pinned code
head, checks the exact asset universe and GitHub digests, anonymously downloads
the package receipt and every part, reconstructs and deeply verifies the
archive, and live-probes the embedded upstream publication. Supplying both
local upstream receipt paths is optional at this stage; if either is supplied,
both are required and must exactly match the embedded binding.

The self-hashed `downstream_publication_receipt.json` records the immutable
release identity, anonymous downloads, verifier identity, scientific status,
locks, terminal digest, and complete upstream binding. Its output path must not
already exist.

## Tests

```powershell
python -m unittest -v test_crossscan_downstream_publication.py
```
