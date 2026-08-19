param(
    [Parameter(Mandatory = $true)]
    [string]$Confirmation,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExpectedCodeHead,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{64}$')]
    [string]$ExpectedTerminalContentSHA256,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{64}$')]
    [string]$ExpectedUpstreamPublicationContentSHA256,
    [string]$Python = "C:\Users\PC\miniconda3\envs\vesuvius\python.exe",
    [string]$DownstreamDir,
    [string]$PublicationRoot,
    [string]$ModelPackageReceipt,
    [string]$ModelPublicationReceipt,
    [ValidateRange(1, 20)]
    [int]$PropagationAttempts = 8,
    [ValidateRange(0, 60)]
    [int]$PropagationDelaySeconds = 5
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

# Exit codes intentionally distinguish failures that are safe to rerun from an
# immutable publication whose anonymous receipt still needs to be recovered.
$exitPreflightMismatch = 10
$exitPublisherBusy = 11
$exitDraftPreserved = 20
$exitPublishedReadbackPending = 30
$script:PublisherExitCode = $exitPreflightMismatch

# Deliberately separate from the model publisher. This script consumes an
# already sealed downstream tree and an already verified immutable model.
$repo = [IO.Path]::GetFullPath($PSScriptRoot)
$workspace = [IO.Path]::GetFullPath((Join-Path $repo "..\.."))
$tool = Join-Path $repo "crossscan_downstream_publication.py"
if (-not $DownstreamDir) {
    $DownstreamDir = Join-Path $workspace "data\crossscan_scrollfiesta_v4\downstream"
}
if (-not $PublicationRoot) {
    $PublicationRoot = Join-Path $workspace "data\crossscan_scrollfiesta_v4_downstream_publication"
}
if (-not $ModelPackageReceipt) {
    $ModelPackageReceipt = Join-Path $workspace "data\crossscan_release_v4_publication\package_receipt.json"
}
if (-not $ModelPublicationReceipt) {
    $ModelPublicationReceipt = Join-Path $workspace "data\crossscan_release_v4_publication\publication_receipt.json"
}
$PublicationRoot = [IO.Path]::GetFullPath($PublicationRoot)
$packageReceipt = Join-Path $PublicationRoot "downstream_package_receipt.json"
$publicationReceipt = Join-Path $PublicationRoot "downstream_publication_receipt.json"
$notesPath = Join-Path $PublicationRoot "DOWNSTREAM_RELEASE_NOTES.md"
$repository = "TAUIL-Abd-Elilah/vesuvius-repro"
$expectedOrigin = "https://github.com/TAUIL-Abd-Elilah/vesuvius-repro.git"
$approvedCodeRef = "refs/heads/physical-crossscan-release-tooling"

function Get-ByteSHA256 {
    param(
        [Parameter(Mandatory = $true)]
        [byte[]]$Bytes
    )

    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($algorithm.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant()
    } finally {
        $algorithm.Dispose()
    }
}

function Get-TextSHA256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return Get-ByteSHA256 -Bytes ([Text.Encoding]::UTF8.GetBytes($Value))
}

function Read-OpenFileBytes {
    param(
        [Parameter(Mandatory = $true)]
        [IO.FileStream]$Stream
    )

    if ($Stream.Length -gt [int]::MaxValue) {
        throw "package receipt is unexpectedly large"
    }
    $buffer = [byte[]]::new([int]$Stream.Length)
    $Stream.Position = 0
    $offset = 0
    while ($offset -lt $buffer.Length) {
        $read = $Stream.Read($buffer, $offset, $buffer.Length - $offset)
        if ($read -le 0) {
            throw "package receipt changed while it was being captured"
        }
        $offset += $read
    }
    return ,$buffer
}

function Assert-PackageReceiptUnchanged {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [int64]$ExpectedBytes,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedSHA256
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "captured downstream package receipt disappeared"
    }
    $current = [IO.File]::ReadAllBytes($Path)
    if ([int64]$current.Length -ne $ExpectedBytes -or
            (Get-ByteSHA256 -Bytes $current) -cne $ExpectedSHA256) {
        throw "downstream package receipt identity changed during publication"
    }
}

function Get-GitHubReleases {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repository,
        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    # --slurp makes all paginated response arrays one strict JSON document.
    $rawPages = @(& gh api --paginate --slurp "repos/$Repository/releases")
    $queryExit = $LASTEXITCODE
    if ($queryExit -ne 0) {
        throw $FailureMessage
    }
    $payload = [string]::Join([char]10, [string[]]$rawPages)
    if ([string]::IsNullOrWhiteSpace($payload)) {
        throw $FailureMessage
    }
    $pages = ConvertFrom-Json -InputObject $payload
    return @(
        foreach ($page in @($pages)) {
            foreach ($release in @($page)) {
                $release
            }
        }
    )
}

function Get-GitHubReleaseById {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repository,
        [Parameter(Mandatory = $true)]
        [int64]$ReleaseId,
        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    $rawRelease = @(& gh api "repos/$Repository/releases/$ReleaseId")
    $queryExit = $LASTEXITCODE
    if ($queryExit -ne 0) {
        throw $FailureMessage
    }
    $payload = [string]::Join([char]10, [string[]]$rawRelease)
    if ([string]::IsNullOrWhiteSpace($payload)) {
        throw $FailureMessage
    }
    return ConvertFrom-Json -InputObject $payload
}

function Assert-ApprovedCodeIdentity {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CodeHead,
        [Parameter(Mandatory = $true)]
        [string]$Origin,
        [Parameter(Mandatory = $true)]
        [string]$ApprovedRef
    )

    $status = @(& git status --porcelain=v1)
    $statusExit = $LASTEXITCODE
    $localHeadRaw = @(& git rev-parse HEAD)
    $headExit = $LASTEXITCODE
    $originRaw = @(& git remote get-url origin)
    $originExit = $LASTEXITCODE
    if ($statusExit -ne 0 -or $headExit -ne 0 -or $originExit -ne 0 -or
            $status.Count -ne 0 -or $localHeadRaw.Count -ne 1 -or
            $originRaw.Count -ne 1) {
        throw "downstream publication worktree is unavailable or dirty"
    }
    $localHead = ([string]$localHeadRaw[0]).Trim()
    $originValue = ([string]$originRaw[0]).Trim()
    if ($localHead -cne $CodeHead -or $originValue -cne $Origin) {
        throw "local code identity differs from the independently approved public head"
    }

    $approvedLines = @(& git ls-remote origin $ApprovedRef)
    $approvedExit = $LASTEXITCODE
    if ($approvedExit -ne 0 -or $approvedLines.Count -ne 1) {
        throw "approved public code ref could not be resolved uniquely: $ApprovedRef"
    }
    $fields = @(([string]$approvedLines[0]).Trim() -split "\s+")
    if ($fields.Count -ne 2 -or $fields[0] -cne $CodeHead -or
            $fields[1] -cne $ApprovedRef) {
        throw "approved public code ref does not pin the independently expected head"
    }
}

function Assert-ReleaseStaticMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Release,
        [Parameter(Mandatory = $true)]
        [int64]$ReleaseId,
        [Parameter(Mandatory = $true)]
        [string]$Tag,
        [Parameter(Mandatory = $true)]
        [string]$CodeHead,
        [Parameter(Mandatory = $true)]
        [string]$Title,
        [Parameter(Mandatory = $true)]
        [string]$Body
    )

    if ([int64]$Release.id -ne $ReleaseId -or $ReleaseId -le 0 -or
            $Release.prerelease -ne $false -or
            $Release.tag_name -cne $Tag -or
            $Release.target_commitish -cne $CodeHead -or
            $Release.name -cne $Title -or
            [string]$Release.body -cne $Body) {
        throw "downstream release metadata differs from the exact publication contract"
    }
}

function Test-ExactDraftState {
    param([Parameter(Mandatory = $true)][object]$Release)
    return $Release.draft -eq $true -and $Release.immutable -eq $false
}

function Test-ExactPublishedState {
    param([Parameter(Mandatory = $true)][object]$Release)
    return $Release.draft -eq $false -and $Release.immutable -eq $true
}

function Assert-ExactAssetUniverse {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Release,
        [Parameter(Mandatory = $true)]
        [Collections.Generic.Dictionary[string, object]]$ExpectedAssets,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $assets = @($Release.assets)
    if ($assets.Count -ne $ExpectedAssets.Count) {
        throw "$Context asset universe is not exactly parts plus package receipt"
    }
    $verifiedNames = [Collections.Generic.Dictionary[string, object]]::new(
        [StringComparer]::Ordinal
    )
    foreach ($asset in $assets) {
        $name = [string]$asset.name
        if (-not $ExpectedAssets.ContainsKey($name) -or $verifiedNames.ContainsKey($name)) {
            throw "$Context contains an unexpected, wrong-case, or duplicate asset: $name"
        }
        $expected = $ExpectedAssets[$name]
        if ($name -cne [string]$expected.name -or [int64]$asset.id -le 0 -or
                $asset.state -cne "uploaded" -or
                [int64]$asset.size -ne [int64]$expected.bytes -or
                [string]$asset.digest -cne [string]$expected.digest) {
            throw "$Context server-side asset identity differs: $name"
        }
        $verifiedNames.Add($name, $true)
    }
}

function Get-ResumableDraftAssets {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Release,
        [Parameter(Mandatory = $true)]
        [Collections.Generic.Dictionary[string, object]]$ExpectedAssets
    )

    $seen = [Collections.Generic.Dictionary[string, object]]::new(
        [StringComparer]::Ordinal
    )
    $observed = [Collections.Generic.Dictionary[string, object]]::new(
        [StringComparer]::Ordinal
    )
    foreach ($asset in @($Release.assets)) {
        $name = [string]$asset.name
        if (-not $ExpectedAssets.ContainsKey($name) -or $observed.ContainsKey($name)) {
            throw "downstream draft contains an unexpected, wrong-case, or duplicate asset: $name"
        }
        $observed.Add($name, $true)
        $expected = $ExpectedAssets[$name]
        if ($name -cne [string]$expected.name -or [int64]$asset.id -le 0) {
            throw "existing downstream draft asset is not safely resumable: $name"
        }
        if ($asset.state -ceq "uploaded" -and
                [int64]$asset.size -eq [int64]$expected.bytes -and
                [string]$asset.digest -ceq [string]$expected.digest) {
            $seen.Add($name, $true)
            continue
        }
        if ($asset.state -ceq "starter") {
            # A starter can belong to another live uploader. GitHub exposes no
            # ownership/lease proof here, so never delete it automatically.
            throw "downstream draft contains a live or unproven-stale starter asset; no asset was deleted: $name"
        }
        throw "existing downstream draft asset is not safely resumable: $name"
    }
    return ,$seen
}

function Get-ExactTagState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Tag,
        [Parameter(Mandatory = $true)]
        [string]$CodeHead
    )

    $lines = @(& git ls-remote origin "refs/tags/$Tag")
    $queryExit = $LASTEXITCODE
    if ($queryExit -ne 0) {
        return [pscustomobject]@{ QuerySucceeded = $false; Present = $false }
    }
    if ($lines.Count -eq 0) {
        return [pscustomobject]@{ QuerySucceeded = $true; Present = $false }
    }
    if ($lines.Count -ne 1) {
        throw "public downstream tag is not unique: $Tag"
    }
    $fields = @(([string]$lines[0]).Trim() -split "\s+")
    if ($fields.Count -ne 2 -or $fields[0] -cne $CodeHead -or
            $fields[1] -cne "refs/tags/$Tag") {
        throw "public downstream tag does not resolve exactly to the approved code head"
    }
    return [pscustomobject]@{ QuerySucceeded = $true; Present = $true }
}

function Wait-ExactPublicTag {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Tag,
        [Parameter(Mandatory = $true)]
        [string]$CodeHead,
        [Parameter(Mandatory = $true)]
        [int]$Attempts,
        [Parameter(Mandatory = $true)]
        [int]$DelaySeconds
    )

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        $state = Get-ExactTagState -Tag $Tag -CodeHead $CodeHead
        if ($state.QuerySucceeded -and $state.Present) {
            return
        }
        if ($attempt -lt $Attempts -and $DelaySeconds -gt 0) {
            Start-Sleep -Seconds $DelaySeconds
        }
    }
    throw "published downstream tag did not propagate within $Attempts bounded attempts"
}

function Get-CanonicalReleaseBody {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Package,
        [Parameter(Mandatory = $true)]
        [string]$ScientificStatus,
        [Parameter(Mandatory = $true)]
        [string]$CodeHead,
        [Parameter(Mandatory = $true)]
        [string]$TerminalDigest,
        [Parameter(Mandatory = $true)]
        [string]$UpstreamPublicationDigest
    )

    $claim = if ($ScientificStatus -ceq "PASS") {
        "Bounded untouched-PHerc0139 probability-to-ScrollFiesta improvement."
    } else {
        "Bounded negative result; no downstream-improvement claim is authorized."
    }
    $lines = @(
        "# Cross-scan v4 downstream result",
        "",
        $claim,
        "",
        "- Scientific status: $ScientificStatus",
        "- Downstream tooling code head: $CodeHead",
        "- Terminal receipt content SHA-256: $TerminalDigest",
        "- Downstream lock content SHA-256: $($Package.downstream.downstream_lock_content_sha256)",
        "- Metric lock content SHA-256: $($Package.downstream.metric_lock_content_sha256)",
        "- Downstream archive SHA-256: $($Package.archive.sha256)",
        "- Downstream package content SHA-256: $($Package.content_sha256)",
        "- Upstream model tag: $($Package.upstream_model_publication.tag)",
        "- Upstream model code head: $($Package.upstream_model_publication.code_head)",
        "- Upstream model manifest content SHA-256: $($Package.upstream_model_publication.release_manifest_content_sha256)",
        "- Upstream model package content SHA-256: $($Package.upstream_model_publication.release_package_content_sha256)",
        "- Upstream model publication receipt content SHA-256: $UpstreamPublicationDigest",
        "",
        "The ordered parts reconstruct the exact terminal-bound regular-file universe.",
        "Scientific FAIL is a valid sealed result and must not be selectively rerun."
    )
    return [string]::Join([char]10, [string[]]$lines) + [char]10
}

$publisherMutex = $null
$publisherMutexHeld = $false
$packageHandle = $null
$snapshotRoot = $null

try {
    foreach ($required in @(
        $repo, $tool, $Python, $DownstreamDir, $PublicationRoot,
        $packageReceipt, $ModelPackageReceipt, $ModelPublicationReceipt
    )) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "required downstream publication input is missing: $required"
        }
    }
    if (Test-Path -LiteralPath $publicationReceipt) {
        throw "downstream publication receipt already exists: $publicationReceipt"
    }

    # A named mutex is held through final receipt placement. Unlike a stale
    # lock file it is released by the OS if the publisher process terminates.
    # The release tag is derived from the terminal digest, so every contender
    # for that tag must share one mutex even if an independently supplied code
    # head differs or the approved branch advances between invocations.
    $lockKey = "$repository|$ExpectedTerminalContentSHA256"
    $mutexPrefix = if ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) {
        "Global\"
    } else {
        ""
    }
    $mutexName = $mutexPrefix + "CrossscanV4DownstreamPublisher_$(Get-TextSHA256 -Value $lockKey)"
    $publisherMutex = [Threading.Mutex]::new($false, $mutexName)
    try {
        $publisherMutexHeld = $publisherMutex.WaitOne(0)
    } catch [Threading.AbandonedMutexException] {
        $publisherMutexHeld = $true
    }
    if (-not $publisherMutexHeld) {
        $script:PublisherExitCode = $exitPublisherBusy
        throw "another downstream publisher holds the single-publisher lock"
    }

    # Keep a read-only, non-delete-shared handle open on Windows and also hash
    # the exact captured bytes. The path is re-read at every mutation boundary.
    $packageHandle = [IO.File]::Open(
        $packageReceipt,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read
    )
    $packageBytes = Read-OpenFileBytes -Stream $packageHandle
    $packageByteCount = [int64]$packageBytes.Length
    $packageByteSHA256 = Get-ByteSHA256 -Bytes $packageBytes
    $packageText = [Text.UTF8Encoding]::new($false, $true).GetString($packageBytes)
    $package = ConvertFrom-Json -InputObject $packageText

    if ([string]$package.schema_version -cne "crossscan-scrollfiesta-downstream-package-v1" -or
            [string]$package.status -cne "PASS") {
        throw "downstream package receipt header is invalid"
    }
    $archive = Join-Path $PublicationRoot ([string]$package.archive.path)
    $codeHead = [string]$package.code_head
    $scientificStatus = [string]$package.downstream.scientific_status
    $terminalDigest = [string]$package.downstream.terminal_receipt.content_sha256
    $upstreamPublicationDigest = [string]$package.upstream_model_publication.publication_receipt.content_sha256
    $tag = [string]$package.publication_tag
    if ($codeHead -notmatch '^[0-9a-f]{40}$' -or
            $terminalDigest -notmatch '^[0-9a-f]{64}$' -or
            $upstreamPublicationDigest -notmatch '^[0-9a-f]{64}$') {
        throw "downstream package contains an invalid publication identity"
    }
    if ($scientificStatus -cne "PASS" -and $scientificStatus -cne "FAIL") {
        throw "scientific status must be exactly PASS or FAIL"
    }
    $expectedTag = "crossscan-v4-downstream-" + $terminalDigest.Substring(0, 12)
    $expectedConfirmation = (
        "PUBLISH IMMUTABLE CROSSSCAN V4 DOWNSTREAM " +
        "$scientificStatus $terminalDigest AT $ExpectedCodeHead"
    )
    $releaseTitle = "Cross-scan v4 downstream $scientificStatus ($($terminalDigest.Substring(0, 12)))"
    if ($codeHead -cne $ExpectedCodeHead) {
        throw "package code head differs from the independently expected code head"
    }
    if ($terminalDigest -cne $ExpectedTerminalContentSHA256) {
        throw "package differs from independently expected terminal digest"
    }
    if ($upstreamPublicationDigest -cne $ExpectedUpstreamPublicationContentSHA256) {
        throw "package differs from independently expected upstream publication receipt"
    }
    if ($tag -cne $expectedTag) {
        throw "downstream tag is not derived from the terminal digest"
    }
    if ($Confirmation -cne $expectedConfirmation) {
        throw "exact confirmation required: $expectedConfirmation"
    }

    $notes = Get-CanonicalReleaseBody `
        -Package $package `
        -ScientificStatus $scientificStatus `
        -CodeHead $ExpectedCodeHead `
        -TerminalDigest $terminalDigest `
        -UpstreamPublicationDigest $upstreamPublicationDigest
    $notesBytes = [Text.UTF8Encoding]::new($false).GetBytes($notes)
    if (Test-Path -LiteralPath $notesPath) {
        $currentNotes = [IO.File]::ReadAllBytes($notesPath)
        if ((Get-ByteSHA256 -Bytes $currentNotes) -cne
                (Get-ByteSHA256 -Bytes $notesBytes)) {
            throw "existing downstream notes differ from the canonical package body"
        }
    }

    $verifyArguments = @(
        $tool, "verify",
        "--archive", $archive,
        "--receipt", $packageReceipt,
        "--downstream-dir", $DownstreamDir,
        "--model-package-receipt", $ModelPackageReceipt,
        "--model-publication-receipt", $ModelPublicationReceipt,
        "--expected-terminal-content-sha256", $ExpectedTerminalContentSHA256,
        "--expected-upstream-publication-content-sha256",
        $ExpectedUpstreamPublicationContentSHA256
    )
    & $Python @verifyArguments
    if ($LASTEXITCODE -ne 0) {
        throw "local downstream package verification failed"
    }
    Assert-PackageReceiptUnchanged `
        -Path $packageReceipt `
        -ExpectedBytes $packageByteCount `
        -ExpectedSHA256 $packageByteSHA256

    Set-Location $repo
    Assert-ApprovedCodeIdentity `
        -CodeHead $ExpectedCodeHead `
        -Origin $expectedOrigin `
        -ApprovedRef $approvedCodeRef

    $expectedAssets = [Collections.Generic.Dictionary[string, object]]::new(
        [StringComparer]::Ordinal
    )
    $assetUploadOrder = @()
    $parts = @($package.archive.parts)
    if ($parts.Count -eq 0 -or $parts.Count -gt 999) {
        throw "downstream archive part count is outside the GitHub asset contract"
    }
    foreach ($part in $parts) {
        $name = [string]$part.path
        $partPath = Join-Path $PublicationRoot $name
        if (-not (Test-Path -LiteralPath $partPath -PathType Leaf)) {
            throw "required downstream archive part is missing: $partPath"
        }
        if ($expectedAssets.ContainsKey($name)) {
            throw "duplicate downstream archive asset name: $name"
        }
        $record = [pscustomobject]@{
            name = $name
            path = $partPath
            bytes = [int64]$part.bytes
            digest = "sha256:$([string]$part.sha256)"
        }
        $expectedAssets.Add($name, $record)
        $assetUploadOrder += $record
    }
    $packageAssetName = "downstream_package_receipt.json"
    if ($expectedAssets.ContainsKey($packageAssetName)) {
        throw "downstream package asset name collides with an archive part"
    }
    $packageAsset = [pscustomobject]@{
        name = $packageAssetName
        path = $packageReceipt
        bytes = $packageByteCount
        digest = "sha256:$packageByteSHA256"
    }
    $expectedAssets.Add($packageAssetName, $packageAsset)
    $assetUploadOrder += $packageAsset

    $identityRaw = @(& gh api user)
    $identityExit = $LASTEXITCODE
    if ($identityExit -ne 0) {
        throw "GitHub CLI identity could not be read"
    }
    $identity = ConvertFrom-Json -InputObject ([string]::Join([char]10, [string[]]$identityRaw))
    if ([string]$identity.login -cne "TAUIL-Abd-Elilah") {
        throw "GitHub CLI is not authenticated as TAUIL-Abd-Elilah"
    }
    $repositoryRaw = @(& gh api "repos/$repository")
    $repositoryExit = $LASTEXITCODE
    if ($repositoryExit -ne 0) {
        throw "target GitHub repository could not be read"
    }
    $repositoryValue = ConvertFrom-Json -InputObject (
        [string]::Join([char]10, [string[]]$repositoryRaw)
    )
    if ($repositoryValue.private -ne $false -or
            $repositoryValue.permissions.admin -ne $true) {
        throw "target GitHub repository is private or current account lacks admin permission"
    }

    $existingReleases = @(Get-GitHubReleases `
        -Repository $repository `
        -FailureMessage "could not enumerate existing downstream releases")
    $matchingReleases = @(
        $existingReleases |
            Where-Object { [string]$_.tag_name -ceq $tag }
    )
    if ($matchingReleases.Count -gt 1) {
        throw "multiple GitHub releases use the pinned downstream tag: $tag"
    }

    $draft = $null
    $published = $null
    $readbackOnlyRecovery = $false
    $releaseId = [int64]0
    if ($matchingReleases.Count -eq 1) {
        $releaseId = [int64]$matchingReleases[0].id
        if ($releaseId -le 0) {
            throw "matching downstream release has no valid release id"
        }
        $candidate = Get-GitHubReleaseById `
            -Repository $repository `
            -ReleaseId $releaseId `
            -FailureMessage "could not read the matching downstream release by id"
        Assert-ReleaseStaticMetadata `
            -Release $candidate `
            -ReleaseId $releaseId `
            -Tag $tag `
            -CodeHead $ExpectedCodeHead `
            -Title $releaseTitle `
            -Body $notes
        if (Test-ExactDraftState -Release $candidate) {
            $script:PublisherExitCode = $exitDraftPreserved
            $draft = $candidate
        } elseif (Test-ExactPublishedState -Release $candidate) {
            Assert-ExactAssetUniverse `
                -Release $candidate `
                -ExpectedAssets $expectedAssets `
                -Context "published downstream release"
            $script:PublisherExitCode = $exitPublishedReadbackPending
            $published = $candidate
            $readbackOnlyRecovery = $true
        } else {
            throw "existing downstream release is neither the exact draft nor exact immutable publication"
        }
    }

    $initialTagState = Get-ExactTagState -Tag $tag -CodeHead $ExpectedCodeHead
    if (-not $initialTagState.QuerySucceeded) {
        throw "could not check the public downstream release tag"
    }
    if ($null -eq $published -and $initialTagState.Present) {
        throw "public downstream tag exists without the exact immutable release"
    }
    if ($null -ne $published -and -not $initialTagState.Present) {
        # A just-published tag is allowed to propagate during bounded readback.
    }

    $seenAssets = [Collections.Generic.Dictionary[string, object]]::new(
        [StringComparer]::Ordinal
    )
    if ($null -ne $draft) {
        $seenAssets = Get-ResumableDraftAssets `
            -Release $draft `
            -ExpectedAssets $expectedAssets
    }

    # Close the enumeration-to-mutation window as far as the APIs allow. A
    # mismatch discovered here must leave GitHub completely untouched.
    if ($null -eq $published) {
        Assert-PackageReceiptUnchanged `
            -Path $packageReceipt `
            -ExpectedBytes $packageByteCount `
            -ExpectedSHA256 $packageByteSHA256
        Assert-ApprovedCodeIdentity `
            -CodeHead $ExpectedCodeHead `
            -Origin $expectedOrigin `
            -ApprovedRef $approvedCodeRef
        $gateReleases = @(Get-GitHubReleases `
            -Repository $repository `
            -FailureMessage "could not perform the pre-mutation release recheck")
        $gateMatches = @(
            $gateReleases |
                Where-Object { [string]$_.tag_name -ceq $tag }
        )
        if ($null -eq $draft) {
            if ($gateMatches.Count -ne 0) {
                throw "downstream release appeared before draft creation"
            }
        } else {
            if ($gateMatches.Count -ne 1 -or [int64]$gateMatches[0].id -ne $releaseId) {
                throw "resumable downstream draft is no longer unique"
            }
            $draft = Get-GitHubReleaseById `
                -Repository $repository `
                -ReleaseId $releaseId `
                -FailureMessage "could not perform the pre-mutation draft recheck"
            Assert-ReleaseStaticMetadata `
                -Release $draft `
                -ReleaseId $releaseId `
                -Tag $tag `
                -CodeHead $ExpectedCodeHead `
                -Title $releaseTitle `
                -Body $notes
            if (-not (Test-ExactDraftState -Release $draft)) {
                throw "release changed before the first publisher mutation"
            }
            $seenAssets = Get-ResumableDraftAssets `
                -Release $draft `
                -ExpectedAssets $expectedAssets
        }
        $gateTag = Get-ExactTagState -Tag $tag -CodeHead $ExpectedCodeHead
        if (-not $gateTag.QuerySucceeded -or $gateTag.Present) {
            throw "public downstream tag appeared before the first publisher mutation"
        }
    }

    # No mismatch above this point caused a local or GitHub write. Snapshot the
    # captured package and canonical body for all later commands.
    $snapshotName = ".crossscan-publish-$PID-$([Guid]::NewGuid().ToString('N'))"
    $snapshotRoot = Join-Path $PublicationRoot $snapshotName
    [void][IO.Directory]::CreateDirectory($snapshotRoot)
    $snapshotPackageReceipt = Join-Path $snapshotRoot $packageAssetName
    $snapshotNotes = Join-Path $snapshotRoot "DOWNSTREAM_RELEASE_NOTES.md"
    [IO.File]::WriteAllBytes($snapshotPackageReceipt, $packageBytes)
    [IO.File]::WriteAllBytes($snapshotNotes, $notesBytes)
    $packageAsset.path = $snapshotPackageReceipt

    if ($null -eq $published) {
        $immutableRaw = @(& gh api "repos/$repository/immutable-releases")
        $immutableExit = $LASTEXITCODE
        if ($immutableExit -ne 0) {
            throw "could not read immutable-release setting"
        }
        $immutable = ConvertFrom-Json -InputObject (
            [string]::Join([char]10, [string[]]$immutableRaw)
        )
        if ($immutable.enabled -ne $true) {
            $script:PublisherExitCode = $exitDraftPreserved
            $null = @(& gh api --method PUT "repos/$repository/immutable-releases")
            if ($LASTEXITCODE -ne 0) {
                throw "could not enable GitHub immutable releases"
            }
        }
        $immutableRaw = @(& gh api "repos/$repository/immutable-releases")
        $immutableExit = $LASTEXITCODE
        if ($immutableExit -ne 0) {
            throw "could not re-read immutable-release setting"
        }
        $immutable = ConvertFrom-Json -InputObject (
            [string]::Join([char]10, [string[]]$immutableRaw)
        )
        if ($immutable.enabled -ne $true) {
            throw "GitHub immutable releases are not enabled"
        }

        if ($null -eq $draft) {
            Assert-PackageReceiptUnchanged `
                -Path $packageReceipt `
                -ExpectedBytes $packageByteCount `
                -ExpectedSHA256 $packageByteSHA256
            $script:PublisherExitCode = $exitDraftPreserved
            $createArguments = @(
                "release", "create", $tag,
                "--repo", $repository,
                "--target", $ExpectedCodeHead,
                "--title", $releaseTitle,
                "--notes-file", $snapshotNotes,
                "--draft"
            )
            $null = @(& gh @createArguments)
            if ($LASTEXITCODE -ne 0) {
                throw "draft creation failed; preserve any matching downstream release and rerun"
            }
            $createdReleases = @(Get-GitHubReleases `
                -Repository $repository `
                -FailureMessage "draft created, but downstream releases could not be re-enumerated")
            $createdMatches = @(
                $createdReleases |
                    Where-Object { [string]$_.tag_name -ceq $tag }
            )
            if ($createdMatches.Count -ne 1) {
                throw "draft created, but its unique downstream release id could not be resolved"
            }
            $releaseId = [int64]$createdMatches[0].id
            $draft = Get-GitHubReleaseById `
                -Repository $repository `
                -ReleaseId $releaseId `
                -FailureMessage "could not read the newly created downstream draft"
            Assert-ReleaseStaticMetadata `
                -Release $draft `
                -ReleaseId $releaseId `
                -Tag $tag `
                -CodeHead $ExpectedCodeHead `
                -Title $releaseTitle `
                -Body $notes
            if (-not (Test-ExactDraftState -Release $draft)) {
                throw "new downstream release is not the exact resumable draft"
            }
            $seenAssets = Get-ResumableDraftAssets `
                -Release $draft `
                -ExpectedAssets $expectedAssets
        }

        foreach ($upload in $assetUploadOrder) {
            if ($seenAssets.ContainsKey([string]$upload.name)) {
                continue
            }
            if ([string]$upload.name -ceq $packageAssetName) {
                Assert-PackageReceiptUnchanged `
                    -Path $packageReceipt `
                    -ExpectedBytes $packageByteCount `
                    -ExpectedSHA256 $packageByteSHA256
            }
            $encodedName = [Uri]::EscapeDataString([string]$upload.name)
            $uploadUrl = (
                "https://uploads.github.com/repos/$repository/releases/$releaseId/assets" +
                "?name=$encodedName"
            )
            $null = @(& gh api --method POST `
                -H "Content-Type: application/octet-stream" `
                --input ([string]$upload.path) `
                $uploadUrl)
            if ($LASTEXITCODE -ne 0) {
                throw "asset upload failed; preserve downstream draft $releaseId and rerun: $($upload.name)"
            }
        }

        $uploadedDraft = Get-GitHubReleaseById `
            -Repository $repository `
            -ReleaseId $releaseId `
            -FailureMessage "could not re-read the uploaded downstream draft by release id"
        Assert-ReleaseStaticMetadata `
            -Release $uploadedDraft `
            -ReleaseId $releaseId `
            -Tag $tag `
            -CodeHead $ExpectedCodeHead `
            -Title $releaseTitle `
            -Body $notes
        if (-not (Test-ExactDraftState -Release $uploadedDraft)) {
            throw "uploaded downstream release is not the exact draft"
        }
        Assert-ExactAssetUniverse `
            -Release $uploadedDraft `
            -ExpectedAssets $expectedAssets `
            -Context "uploaded downstream draft"

        # Recheck every independently pinned identity before the final by-id
        # fetch. Nothing except the exact PATCH occurs after that final fetch.
        Assert-PackageReceiptUnchanged `
            -Path $packageReceipt `
            -ExpectedBytes $packageByteCount `
            -ExpectedSHA256 $packageByteSHA256
        Assert-ApprovedCodeIdentity `
            -CodeHead $ExpectedCodeHead `
            -Origin $expectedOrigin `
            -ApprovedRef $approvedCodeRef
        $prePublishTag = Get-ExactTagState -Tag $tag -CodeHead $ExpectedCodeHead
        if (-not $prePublishTag.QuerySucceeded -or $prePublishTag.Present) {
            throw "public downstream tag appeared before the draft publication transition"
        }
        $finalReleases = @(Get-GitHubReleases `
            -Repository $repository `
            -FailureMessage "could not perform the final unique-release recheck")
        $finalMatches = @(
            $finalReleases |
                Where-Object { [string]$_.tag_name -ceq $tag }
        )
        if ($finalMatches.Count -ne 1 -or [int64]$finalMatches[0].id -ne $releaseId) {
            throw "final downstream draft is no longer the unique tagged release"
        }

        # Exact final draft recheck immediately before the irreversible PATCH.
        $finalDraft = Get-GitHubReleaseById `
            -Repository $repository `
            -ReleaseId $releaseId `
            -FailureMessage "could not perform the exact final draft recheck"
        Assert-ReleaseStaticMetadata `
            -Release $finalDraft `
            -ReleaseId $releaseId `
            -Tag $tag `
            -CodeHead $ExpectedCodeHead `
            -Title $releaseTitle `
            -Body $notes
        if (-not (Test-ExactDraftState -Release $finalDraft)) {
            throw "release changed before the immutable publication transition"
        }
        Assert-ExactAssetUniverse `
            -Release $finalDraft `
            -ExpectedAssets $expectedAssets `
            -Context "final downstream draft"

        $null = @(& gh api --method PATCH `
            "repos/$repository/releases/$releaseId" `
            -F draft=false)
        $publishExit = $LASTEXITCODE
        $script:PublisherExitCode = $exitPublishedReadbackPending

        # A nonzero client exit may still mean GitHub applied the PATCH. Always
        # recover by authoritative id readback before deciding the exit class.
        $published = $null
        $lastPublishReadError = $null
        for ($attempt = 1; $attempt -le $PropagationAttempts; $attempt++) {
            try {
                $candidate = Get-GitHubReleaseById `
                    -Repository $repository `
                    -ReleaseId $releaseId `
                    -FailureMessage "could not read release state after publication PATCH"
                Assert-ReleaseStaticMetadata `
                    -Release $candidate `
                    -ReleaseId $releaseId `
                    -Tag $tag `
                    -CodeHead $ExpectedCodeHead `
                    -Title $releaseTitle `
                    -Body $notes
                if (Test-ExactPublishedState -Release $candidate) {
                    Assert-ExactAssetUniverse `
                        -Release $candidate `
                        -ExpectedAssets $expectedAssets `
                        -Context "published downstream release"
                    $published = $candidate
                    break
                }
                if (-not (Test-ExactDraftState -Release $candidate)) {
                    throw "release entered an unexpected state after publication PATCH"
                }
            } catch {
                $lastPublishReadError = $_.Exception.Message
            }
            if ($attempt -lt $PropagationAttempts -and $PropagationDelaySeconds -gt 0) {
                Start-Sleep -Seconds $PropagationDelaySeconds
            }
        }
        if ($null -eq $published) {
            throw "publication PATCH returned $publishExit but its irreversible outcome could not be confirmed: $lastPublishReadError"
        }
    }

    # Both a freshly published release and an exact pre-existing immutable
    # release converge here. This path performs only public/read-only network
    # operations and local receipt creation.
    Wait-ExactPublicTag `
        -Tag $tag `
        -CodeHead $ExpectedCodeHead `
        -Attempts $PropagationAttempts `
        -DelaySeconds $PropagationDelaySeconds

    $verifiedAttemptPath = $null
    for ($attempt = 1; $attempt -le $PropagationAttempts; $attempt++) {
        $attemptOutput = Join-Path $snapshotRoot "public-readback-$attempt.json"
        $publicArguments = @(
            $tool, "verify-public",
            "--package-receipt", $snapshotPackageReceipt,
            "--repository", $repository,
            "--tag", $tag,
            "--output", $attemptOutput,
            "--expected-code-head", $ExpectedCodeHead,
            "--expected-terminal-content-sha256", $ExpectedTerminalContentSHA256,
            "--expected-upstream-publication-content-sha256",
            $ExpectedUpstreamPublicationContentSHA256
        )
        & $Python @publicArguments
        $readbackExit = $LASTEXITCODE
        if ($readbackExit -eq 0 -and (Test-Path -LiteralPath $attemptOutput -PathType Leaf)) {
            $verifiedAttemptPath = $attemptOutput
            break
        }
        if ($attempt -lt $PropagationAttempts -and $PropagationDelaySeconds -gt 0) {
            Start-Sleep -Seconds $PropagationDelaySeconds
        }
    }
    if ($null -eq $verifiedAttemptPath) {
        throw "release is immutable, but logged-out downstream readback failed after $PropagationAttempts bounded attempts"
    }
    if (Test-Path -LiteralPath $publicationReceipt) {
        throw "publication receipt appeared concurrently; refusing to replace it"
    }
    Move-Item -LiteralPath $verifiedAttemptPath -Destination $publicationReceipt

    $verified = Get-Content -Raw -LiteralPath $publicationReceipt | ConvertFrom-Json
    $script:PublisherExitCode = 0
    [pscustomobject]@{
        Status = $verified.status
        ScientificStatus = $verified.scientific_status
        Tag = $tag
        CodeHead = $ExpectedCodeHead
        TerminalContentSHA256 = $terminalDigest
        ReleaseURL = $verified.release.html_url
        PublicationReceipt = $publicationReceipt
        Immutable = $verified.release.immutable
        RecoveryMode = $readbackOnlyRecovery
    } | Format-List
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit $script:PublisherExitCode
} finally {
    if ($null -ne $packageHandle) {
        $packageHandle.Dispose()
    }
    if ($null -ne $snapshotRoot -and (Test-Path -LiteralPath $snapshotRoot)) {
        $resolvedRoot = [IO.Path]::GetFullPath($snapshotRoot)
        $expectedParent = [IO.Path]::GetFullPath($PublicationRoot).TrimEnd(
            [IO.Path]::DirectorySeparatorChar,
            [IO.Path]::AltDirectorySeparatorChar
        ) + [IO.Path]::DirectorySeparatorChar
        if ($resolvedRoot.StartsWith($expectedParent, [StringComparison]::OrdinalIgnoreCase) -and
                [IO.Path]::GetFileName($resolvedRoot).StartsWith(
                    ".crossscan-publish-",
                    [StringComparison]::Ordinal
                )) {
            Remove-Item -LiteralPath $resolvedRoot -Recurse -Force
        }
    }
    if ($publisherMutexHeld -and $null -ne $publisherMutex) {
        $publisherMutex.ReleaseMutex()
    }
    if ($null -ne $publisherMutex) {
        $publisherMutex.Dispose()
    }
}
