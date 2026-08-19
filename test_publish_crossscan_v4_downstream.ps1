$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "test_publish_crossscan_v4_downstream.ps1 requires Windows"
}

$publisher = (Resolve-Path (Join-Path $PSScriptRoot "publish_crossscan_v4_downstream.ps1")).Path
$pwsh = (Get-Process -Id $PID).Path
$testRoot = Join-Path ([IO.Path]::GetTempPath()) (
    "crossscan-publisher-tests-" + [Guid]::NewGuid().ToString("N")
)
[void][IO.Directory]::CreateDirectory($testRoot)

function Assert-True {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$Condition,
        [Parameter(Mandatory = $true)]
        [string]$Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Value
    )
    [IO.File]::WriteAllText($Path, $Value, [Text.UTF8Encoding]::new($false))
}

function Get-SHA256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-StringSHA256 {
    param([Parameter(Mandatory = $true)][string]$Value)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $algorithm.ComputeHash([Text.Encoding]::UTF8.GetBytes($Value))
        return ([BitConverter]::ToString($digest)).Replace("-", "").ToLowerInvariant()
    } finally {
        $algorithm.Dispose()
    }
}

function Get-CanonicalBody {
    param(
        [Parameter(Mandatory = $true)][object]$Package,
        [Parameter(Mandatory = $true)][string]$CodeHead
    )
    $status = [string]$Package.downstream.scientific_status
    $claim = if ($status -ceq "PASS") {
        "Bounded untouched-PHerc0139 probability-to-ScrollFiesta improvement."
    } else {
        "Bounded negative result; no downstream-improvement claim is authorized."
    }
    $lines = @(
        "# Cross-scan v4 downstream result",
        "",
        $claim,
        "",
        "- Scientific status: $status",
        "- Downstream tooling code head: $CodeHead",
        "- Terminal receipt content SHA-256: $($Package.downstream.terminal_receipt.content_sha256)",
        "- Downstream lock content SHA-256: $($Package.downstream.downstream_lock_content_sha256)",
        "- Metric lock content SHA-256: $($Package.downstream.metric_lock_content_sha256)",
        "- Downstream archive SHA-256: $($Package.archive.sha256)",
        "- Downstream package content SHA-256: $($Package.content_sha256)",
        "- Upstream model tag: $($Package.upstream_model_publication.tag)",
        "- Upstream model code head: $($Package.upstream_model_publication.code_head)",
        "- Upstream model manifest content SHA-256: $($Package.upstream_model_publication.release_manifest_content_sha256)",
        "- Upstream model package content SHA-256: $($Package.upstream_model_publication.release_package_content_sha256)",
        "- Upstream model publication receipt content SHA-256: $($Package.upstream_model_publication.publication_receipt.content_sha256)",
        "",
        "The ordered parts reconstruct the exact terminal-bound regular-file universe.",
        "Scientific FAIL is a valid sealed result and must not be selectively rerun."
    )
    return [string]::Join([char]10, [string[]]$lines) + [char]10
}

function New-Fixture {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [ValidateSet("none", "draft", "published")][string]$Phase = "draft",
        [ValidateSet("none", "wrong-case", "starter")][string]$AssetFault = "none",
        [bool]$PatchClientError = $false,
        [int]$TagDelay = 0,
        [int]$CdnFailures = 0,
        [int]$MutateAtGet = 0,
        [bool]$WrongApprovedRef = $false
    )

    $root = Join-Path $testRoot $Name
    $publicationRoot = Join-Path $root "publication"
    $downstream = Join-Path $root "downstream"
    $bin = Join-Path $root "bin"
    foreach ($directory in @($root, $publicationRoot, $downstream, $bin)) {
        [void][IO.Directory]::CreateDirectory($directory)
    }

    $codeHead = "a" * 40
    $terminal = Get-StringSHA256 -Value "terminal:$Name"
    $upstreamPublication = "c" * 64
    $partPath = Join-Path $publicationRoot "part-000.tar"
    Write-Utf8NoBom -Path $partPath -Value "sealed archive part`n"
    $partBytes = [int64](Get-Item -LiteralPath $partPath).Length
    $partSHA = Get-SHA256 -Path $partPath
    $package = [ordered]@{
        schema_version = "crossscan-scrollfiesta-downstream-package-v1"
        status = "PASS"
        created_utc = "2026-08-19T00:00:00+00:00"
        code_head = $codeHead
        publication_tag = "crossscan-v4-downstream-$($terminal.Substring(0, 12))"
        upstream_model_publication = [ordered]@{
            tag = "crossscan-v4-model-fixed"
            code_head = "2" * 40
            release_manifest_content_sha256 = "3" * 64
            release_package_content_sha256 = "4" * 64
            publication_receipt = [ordered]@{ content_sha256 = $upstreamPublication }
        }
        downstream = [ordered]@{
            scientific_status = "PASS"
            terminal_receipt = [ordered]@{ content_sha256 = $terminal }
            downstream_lock_content_sha256 = "5" * 64
            metric_lock_content_sha256 = "6" * 64
        }
        archive = [ordered]@{
            path = "downstream.tar"
            sha256 = "7" * 64
            parts = @(
                [ordered]@{
                    path = "part-000.tar"
                    bytes = $partBytes
                    sha256 = $partSHA
                }
            )
        }
        content_sha256 = "8" * 64
    }
    $packagePath = Join-Path $publicationRoot "downstream_package_receipt.json"
    Write-Utf8NoBom `
        -Path $packagePath `
        -Value (($package | ConvertTo-Json -Depth 20) + "`n")
    $packageBytes = [int64](Get-Item -LiteralPath $packagePath).Length
    $packageSHA = Get-SHA256 -Path $packagePath

    $modelPackage = Join-Path $root "model-package.json"
    $modelPublication = Join-Path $root "model-publication.json"
    Write-Utf8NoBom -Path $modelPackage -Value "{}`n"
    Write-Utf8NoBom -Path $modelPublication -Value "{}`n"

    $body = Get-CanonicalBody -Package $package -CodeHead $codeHead
    $title = "Cross-scan v4 downstream PASS ($($terminal.Substring(0, 12)))"
    $assets = @(
        [ordered]@{
            id = 101
            name = "part-000.tar"
            state = "uploaded"
            size = $partBytes
            digest = "sha256:$partSHA"
        },
        [ordered]@{
            id = 102
            name = "downstream_package_receipt.json"
            state = "uploaded"
            size = $packageBytes
            digest = "sha256:$packageSHA"
        }
    )
    if ($AssetFault -ceq "wrong-case") {
        $assets[0].name = "PART-000.tar"
    } elseif ($AssetFault -ceq "starter") {
        $assets[0].state = "starter"
        $assets[0].size = 0
        $assets[0].digest = ""
    }

    $release = $null
    if ($Phase -cne "none") {
        $release = [ordered]@{
            id = 77
            draft = ($Phase -ceq "draft")
            prerelease = $false
            immutable = ($Phase -ceq "published")
            tag_name = $package.publication_tag
            target_commitish = $codeHead
            name = $title
            body = $body
            html_url = "https://github.example/release/77"
            published_at = if ($Phase -ceq "published") { "2026-08-19T00:00:00Z" } else { $null }
            assets = $assets
        }
    }

    $statePath = Join-Path $root "state.json"
    $state = [ordered]@{
        scenario = $Name
        phase = $Phase
        expectedCode = $codeHead
        tag = $package.publication_tag
        title = $title
        body = $body
        release = $release
        patchClientError = $PatchClientError
        tagDelay = $TagDelay
        tagCalls = 0
        cdnFailures = $CdnFailures
        publicCalls = 0
        getCalls = 0
        mutateAtGet = $MutateAtGet
        wrongApprovedRef = $WrongApprovedRef
        ghMutations = @()
        ghCalls = @()
        gitCalls = @()
    }
    Write-Utf8NoBom -Path $statePath -Value ($state | ConvertTo-Json -Depth 30)

    return [pscustomobject]@{
        Root = $root
        PublicationRoot = $publicationRoot
        Downstream = $downstream
        Bin = $bin
        State = $statePath
        CodeHead = $codeHead
        Terminal = $terminal
        UpstreamPublication = $upstreamPublication
        ModelPackage = $modelPackage
        ModelPublication = $modelPublication
        Package = $packagePath
        Confirmation = "PUBLISH IMMUTABLE CROSSSCAN V4 DOWNSTREAM PASS $terminal AT $codeHead"
    }
}

$fakeGh = @'
$ErrorActionPreference = "Stop"
function Load-State { Get-Content -Raw -LiteralPath $env:CROSSSCAN_FAKE_STATE | ConvertFrom-Json }
function Save-State($state) {
    [IO.File]::WriteAllText(
        $env:CROSSSCAN_FAKE_STATE,
        ($state | ConvertTo-Json -Depth 40),
        [Text.UTF8Encoding]::new($false)
    )
}
function Emit($value) { [Console]::Out.WriteLine(($value | ConvertTo-Json -Depth 40 -Compress)) }
$state = Load-State
$all = @($args)
$state.ghCalls = @($state.ghCalls) + ([string]::Join(" ", [string[]]$all))
if ($all.Count -ge 1 -and $all[0] -ceq "release") {
    $state.ghMutations = @($state.ghMutations) + "release create"
    $tag = $all[2]
    $target = $all[[Array]::IndexOf($all, "--target") + 1]
    $title = $all[[Array]::IndexOf($all, "--title") + 1]
    $notes = Get-Content -Raw -LiteralPath $all[[Array]::IndexOf($all, "--notes-file") + 1]
    $state.release = [pscustomobject]@{
        id = 77; draft = $true; prerelease = $false; immutable = $false
        tag_name = $tag; target_commitish = $target; name = $title; body = $notes
        html_url = "https://github.example/release/77"; published_at = $null; assets = @()
    }
    $state.phase = "draft"
    Save-State $state
    [Console]::Out.WriteLine("created")
    exit 0
}
if ($all.Count -lt 2 -or $all[0] -cne "api") { exit 90 }
$method = "GET"
$methodIndex = [Array]::IndexOf($all, "--method")
if ($methodIndex -ge 0) { $method = $all[$methodIndex + 1] }
$uriCandidates = @(
    $all | Where-Object {
        $_ -ceq "user" -or $_ -like "repos/*" -or $_ -like "https://uploads.github.com/*"
    }
)
if ($uriCandidates.Count -eq 0) { exit 89 }
$uri = $uriCandidates[-1]
if ($method -cne "GET") {
    $state.ghMutations = @($state.ghMutations) + "$method $uri"
}
if ($uri -ceq "user") {
    Save-State $state
    Emit ([ordered]@{ login = "TAUIL-Abd-Elilah" })
    exit 0
}
if ($uri -ceq "repos/TAUIL-Abd-Elilah/vesuvius-repro") {
    Save-State $state
    Emit ([ordered]@{ private = $false; permissions = [ordered]@{ admin = $true } })
    exit 0
}
if ($uri -ceq "repos/TAUIL-Abd-Elilah/vesuvius-repro/immutable-releases") {
    Save-State $state
    Emit ([ordered]@{ enabled = $true })
    exit 0
}
if ($all -contains "--slurp" -and $uri -ceq "repos/TAUIL-Abd-Elilah/vesuvius-repro/releases") {
    Save-State $state
    if ($null -eq $state.release) {
        [Console]::Out.WriteLine("[[]]")
    } else {
        [Console]::Out.WriteLine("[[" + ($state.release | ConvertTo-Json -Depth 40 -Compress) + "]]" )
    }
    exit 0
}
if ($uri -match '^repos/TAUIL-Abd-Elilah/vesuvius-repro/releases/[0-9]+$' -and $method -ceq "GET") {
    $state.getCalls = [int]$state.getCalls + 1
    if ([int]$state.mutateAtGet -gt 0 -and [int]$state.getCalls -eq [int]$state.mutateAtGet) {
        $state.release.body = [string]$state.release.body + "tampered`n"
    }
    Save-State $state
    Emit $state.release
    exit 0
}
if ($method -ceq "POST" -and $uri -like 'https://uploads.github.com/*') {
    $name = [Uri]::UnescapeDataString(($uri -split '\?name=', 2)[1])
    $inputIndex = [Array]::IndexOf($all, "--input")
    $inputPath = $all[$inputIndex + 1]
    $item = Get-Item -LiteralPath $inputPath
    $digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $inputPath).Hash.ToLowerInvariant()
    $newAsset = [pscustomobject]@{
        id = 200 + @($state.release.assets).Count
        name = $name; state = "uploaded"; size = [int64]$item.Length; digest = "sha256:$digest"
    }
    $state.release.assets = @($state.release.assets) + $newAsset
    Save-State $state
    Emit $newAsset
    exit 0
}
if ($method -ceq "PATCH" -and $uri -match '/releases/[0-9]+$') {
    $state.release.draft = $false
    $state.release.immutable = $true
    $state.release.published_at = "2026-08-19T00:00:00Z"
    $state.phase = "published"
    Save-State $state
    if ($state.patchClientError -eq $true) {
        [Console]::Error.WriteLine("simulated client-side failure after server apply")
        exit 9
    }
    Emit $state.release
    exit 0
}
Save-State $state
exit 91
'@

$fakeGit = @'
$ErrorActionPreference = "Stop"
function Load-State { Get-Content -Raw -LiteralPath $env:CROSSSCAN_FAKE_STATE | ConvertFrom-Json }
function Save-State($state) {
    [IO.File]::WriteAllText(
        $env:CROSSSCAN_FAKE_STATE,
        ($state | ConvertTo-Json -Depth 40),
        [Text.UTF8Encoding]::new($false)
    )
}
$state = Load-State
$all = @($args)
$state.gitCalls = @($state.gitCalls) + ([string]::Join(" ", [string[]]$all))
if ($all[0] -ceq "status") { Save-State $state; exit 0 }
if ($all[0] -ceq "rev-parse") { Save-State $state; [Console]::Out.WriteLine($state.expectedCode); exit 0 }
if ($all[0] -ceq "remote") {
    Save-State $state
    [Console]::Out.WriteLine("https://github.com/TAUIL-Abd-Elilah/vesuvius-repro.git")
    exit 0
}
if ($all[0] -ceq "ls-remote") {
    $ref = $all[-1]
    if ($ref -ceq "refs/heads/physical-crossscan-release-tooling") {
        $head = if ($state.wrongApprovedRef -eq $true) { "d" * 40 } else { $state.expectedCode }
        Save-State $state
        [Console]::Out.WriteLine("$head`t$ref")
        exit 0
    }
    if ($ref -like "refs/tags/*") {
        $state.tagCalls = [int]$state.tagCalls + 1
        $show = $state.phase -ceq "published" -and [int]$state.tagCalls -gt [int]$state.tagDelay
        Save-State $state
        if ($show) { [Console]::Out.WriteLine("$($state.expectedCode)`t$ref") }
        exit 0
    }
}
Save-State $state
exit 92
'@

$fakePython = @'
$ErrorActionPreference = "Stop"
function Load-State { Get-Content -Raw -LiteralPath $env:CROSSSCAN_FAKE_STATE | ConvertFrom-Json }
function Save-State($state) {
    [IO.File]::WriteAllText(
        $env:CROSSSCAN_FAKE_STATE,
        ($state | ConvertTo-Json -Depth 40),
        [Text.UTF8Encoding]::new($false)
    )
}
$state = Load-State
$all = @($args)
if ($all.Count -lt 2) { exit 93 }
$command = $all[1]
if ($command -ceq "verify") { exit 0 }
if ($command -ceq "verify-public") {
    $state.publicCalls = [int]$state.publicCalls + 1
    $call = [int]$state.publicCalls
    Save-State $state
    if ($call -le [int]$state.cdnFailures) { exit 8 }
    $outputIndex = [Array]::IndexOf($all, "--output")
    $output = $all[$outputIndex + 1]
    $receipt = [ordered]@{
        status = "PUBLIC_LOGGED_OUT_VERIFIED"
        scientific_status = "PASS"
        release = [ordered]@{
            html_url = "https://github.example/release/77"
            immutable = $true
        }
    }
    [IO.File]::WriteAllText(
        $output,
        (($receipt | ConvertTo-Json -Depth 10) + "`n"),
        [Text.UTF8Encoding]::new($false)
    )
    exit 0
}
exit 94
'@

function Install-Fakes {
    param([Parameter(Mandatory = $true)][object]$Fixture)
    Write-Utf8NoBom -Path (Join-Path $Fixture.Bin "fake-gh.ps1") -Value $fakeGh
    Write-Utf8NoBom -Path (Join-Path $Fixture.Bin "fake-git.ps1") -Value $fakeGit
    Write-Utf8NoBom -Path (Join-Path $Fixture.Bin "fake-python.ps1") -Value $fakePython
    $wrapper = @'
@echo off
"%CROSSSCAN_TEST_PWSH%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0SCRIPT" %*
exit /b %ERRORLEVEL%
'@
    Write-Utf8NoBom `
        -Path (Join-Path $Fixture.Bin "gh.cmd") `
        -Value ($wrapper.Replace("SCRIPT", "fake-gh.ps1"))
    Write-Utf8NoBom `
        -Path (Join-Path $Fixture.Bin "git.cmd") `
        -Value ($wrapper.Replace("SCRIPT", "fake-git.ps1"))
    Write-Utf8NoBom `
        -Path (Join-Path $Fixture.Bin "fake-python.cmd") `
        -Value ($wrapper.Replace("SCRIPT", "fake-python.ps1"))
}

function Invoke-Publisher {
    param([Parameter(Mandatory = $true)][object]$Fixture)
    Install-Fakes -Fixture $Fixture
    $stdoutPath = Join-Path $Fixture.Root "stdout.txt"
    $stderrPath = Join-Path $Fixture.Root "stderr.txt"
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $pwsh
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $publisherArguments = @(
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $publisher,
        "-Confirmation", $Fixture.Confirmation,
        "-ExpectedCodeHead", $Fixture.CodeHead,
        "-ExpectedTerminalContentSHA256", $Fixture.Terminal,
        "-ExpectedUpstreamPublicationContentSHA256", $Fixture.UpstreamPublication,
        "-Python", (Join-Path $Fixture.Bin "fake-python.cmd"),
        "-DownstreamDir", $Fixture.Downstream,
        "-PublicationRoot", $Fixture.PublicationRoot,
        "-ModelPackageReceipt", $Fixture.ModelPackage,
        "-ModelPublicationReceipt", $Fixture.ModelPublication,
        "-PropagationAttempts", "5",
        "-PropagationDelaySeconds", "0"
    )
    function Quote-ProcessArgument {
        param([Parameter(Mandatory = $true)][string]$Value)
        if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
            return $Value
        }
        # Windows CommandLineToArgvW quoting: double backslashes before a quote
        # and before the closing quote.
        return '"' + [regex]::Replace($Value, '(\\*)"', '$1$1\"').Replace(
            '\',
            '\'
        ).TrimEnd() + '"'
    }
    $info.Arguments = [string]::Join(
        " ",
        [string[]]@($publisherArguments | ForEach-Object { Quote-ProcessArgument -Value ([string]$_) })
    )
    $info.EnvironmentVariables["CROSSSCAN_FAKE_STATE"] = $Fixture.State
    $info.EnvironmentVariables["CROSSSCAN_TEST_PWSH"] = $pwsh
    $info.EnvironmentVariables["PATH"] = $Fixture.Bin + [IO.Path]::PathSeparator + $env:PATH
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    [void]$process.Start()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    Write-Utf8NoBom -Path $stdoutPath -Value $stdout
    Write-Utf8NoBom -Path $stderrPath -Value $stderr
    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        Stdout = $stdout
        Stderr = $stderr
        State = Get-Content -Raw -LiteralPath $Fixture.State | ConvertFrom-Json
    }
}

$tests = @(
    [pscustomobject]@{
        Name = "wrong-case uploaded asset aborts without mutation"
        Run = {
            $fixture = New-Fixture -Name "wrong-case" -AssetFault "wrong-case"
            $result = Invoke-Publisher -Fixture $fixture
            Assert-True ($result.ExitCode -eq 20) "wrong-case asset should return draft-preserved exit 20"
            Assert-True ($result.Stderr -like "*wrong-case*") "wrong-case failure should be explicit"
            Assert-True (@($result.State.ghMutations).Count -eq 0) "wrong-case asset caused a GitHub mutation"
        }
    },
    [pscustomobject]@{
        Name = "starter asset aborts and is never deleted"
        Run = {
            $fixture = New-Fixture -Name "starter" -AssetFault "starter"
            $result = Invoke-Publisher -Fixture $fixture
            Assert-True ($result.ExitCode -eq 20) "starter should return draft-preserved exit 20"
            Assert-True ($result.Stderr -like "*no asset was deleted*") "starter failure should state no deletion"
            Assert-True (@($result.State.ghMutations).Count -eq 0) "starter caused a GitHub mutation"
        }
    },
    [pscustomobject]@{
        Name = "independent approved ref mismatch is preflight-only"
        Run = {
            $fixture = New-Fixture -Name "wrong-ref" -WrongApprovedRef $true
            $result = Invoke-Publisher -Fixture $fixture
            Assert-True ($result.ExitCode -eq 10) "wrong approved ref should return preflight exit 10"
            Assert-True ($result.Stderr -like "*approved public code ref*") "approved-ref failure should be explicit"
            Assert-True (@($result.State.ghMutations).Count -eq 0) "approved-ref mismatch caused a GitHub mutation"
        }
    },
    [pscustomobject]@{
        Name = "single-publisher lock has a distinct busy exit"
        Run = {
            $fixture = New-Fixture -Name "publisher-busy"
            $lockKey = "TAUIL-Abd-Elilah/vesuvius-repro|$($fixture.Terminal)"
            $mutexName = "Global\CrossscanV4DownstreamPublisher_$(Get-StringSHA256 -Value $lockKey)"
            $mutex = [Threading.Mutex]::new($false, $mutexName)
            $mutexHeld = $mutex.WaitOne(0)
            try {
                Assert-True $mutexHeld "test could not acquire the publisher mutex"
                $result = Invoke-Publisher -Fixture $fixture
                Assert-True ($result.ExitCode -eq 11) "contending publisher should return busy exit 11"
                Assert-True ($result.Stderr -like "*single-publisher lock*") "busy failure should identify the lock"
                Assert-True (@($result.State.ghMutations).Count -eq 0) "busy publisher caused a GitHub mutation"
            } finally {
                if ($mutexHeld) {
                    $mutex.ReleaseMutex()
                }
                $mutex.Dispose()
            }
        }
    },
    [pscustomobject]@{
        Name = "server-applied client-error PATCH recovers by id"
        Run = {
            $fixture = New-Fixture -Name "patch-client-error" -PatchClientError $true
            $result = Invoke-Publisher -Fixture $fixture
            Assert-True ($result.ExitCode -eq 0) "server-applied client-error PATCH should recover: $($result.Stderr)"
            $patches = @($result.State.ghMutations | Where-Object { $_ -like "PATCH *" })
            Assert-True ($patches.Count -eq 1) "expected exactly one publication PATCH"
            Assert-True (-not (@($result.State.ghMutations) -match "DELETE")) "recovery issued DELETE"
            Assert-True ((Test-Path -LiteralPath (Join-Path $fixture.PublicationRoot "downstream_publication_receipt.json"))) "recovery did not place receipt"
        }
    },
    [pscustomobject]@{
        Name = "existing immutable release is readback-only with bounded retries"
        Run = {
            $fixture = New-Fixture `
                -Name "immutable-recovery" `
                -Phase "published" `
                -TagDelay 2 `
                -CdnFailures 2
            $result = Invoke-Publisher -Fixture $fixture
            Assert-True ($result.ExitCode -eq 0) "immutable recovery should succeed: $($result.Stderr)"
            Assert-True (@($result.State.ghMutations).Count -eq 0) "immutable recovery was not readback-only"
            Assert-True ([int]$result.State.publicCalls -eq 3) "CDN readback retry count was not bounded/expected"
            Assert-True ([int]$result.State.tagCalls -ge 3) "tag propagation was not retried"
        }
    },
    [pscustomobject]@{
        Name = "immutable readback exhaustion has a distinct pending exit"
        Run = {
            $fixture = New-Fixture `
                -Name "immutable-readback-pending" `
                -Phase "published" `
                -CdnFailures 99
            $result = Invoke-Publisher -Fixture $fixture
            Assert-True ($result.ExitCode -eq 30) "exhausted immutable readback should return pending exit 30"
            Assert-True ([int]$result.State.publicCalls -eq 5) "public readback exceeded or missed its retry bound"
            Assert-True (@($result.State.ghMutations).Count -eq 0) "failed immutable readback caused a GitHub mutation"
        }
    },
    [pscustomobject]@{
        Name = "final by-id draft recheck blocks changed metadata"
        Run = {
            $fixture = New-Fixture -Name "final-recheck" -MutateAtGet 4
            $result = Invoke-Publisher -Fixture $fixture
            Assert-True ($result.ExitCode -eq 20) "changed final draft should return draft-preserved exit 20"
            Assert-True (-not (@($result.State.ghMutations) -match "PATCH")) "changed final draft was published"
        }
    }
)

$failures = @()
try {
    foreach ($test in $tests) {
        try {
            & $test.Run
            [Console]::Out.WriteLine("PASS  $($test.Name)")
        } catch {
            $failures += "$($test.Name): $($_.Exception.Message)"
            [Console]::Error.WriteLine("FAIL  $($test.Name): $($_.Exception.Message)")
        }
    }
    if ($failures.Count -gt 0) {
        throw ([string]::Join([Environment]::NewLine, [string[]]$failures))
    }
    [Console]::Out.WriteLine("PowerShell publisher tests: $($tests.Count) passed")
} finally {
    $resolved = [IO.Path]::GetFullPath($testRoot)
    $temporary = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    if ($resolved.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -and
            [IO.Path]::GetFileName($resolved).StartsWith(
                "crossscan-publisher-tests-",
                [StringComparison]::Ordinal
            )) {
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}
