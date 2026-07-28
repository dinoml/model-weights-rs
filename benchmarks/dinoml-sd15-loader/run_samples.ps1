#requires -Version 7.2

[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string] $ResultsDirectory,

    [ValidateRange(10, 10000)]
    [int] $Samples = 10,

    [ValidateRange(1, 1024)]
    [int] $Workers = [Environment]::ProcessorCount,

    [ValidateRange(1, 1048576)]
    [int] $DeliveryQueueDepth = 2,

    [ValidateRange(1, 1048576)]
    [int] $DispatchLookahead = [Environment]::ProcessorCount,

    [ValidateRange(1, 9223372036854775807)]
    [long] $SourceBytes = 256MB,

    [ValidateRange(1, 9223372036854775807)]
    [long] $ScratchBytes = 64MB,

    [ValidateRange(1, 9223372036854775807)]
    [long] $PreparedBytes = 512MB,

    [ValidateSet("delivery", "sha256")]
    [string] $Consumption = "sha256",

    [Alias("BenchmarkExecutable")]
    [string] $Executable,

    [string] $CacheDirectory,

    [switch] $ResetPrepared,

    [Alias("Prime")]
    [switch] $PrimePrepared,

    [switch] $SkipValidation,

    [ValidateRange(1, 1000)]
    [int] $PollIntervalMilliseconds = 10,

    [string] $ClipCheckpoint,

    [string] $ClipArtifacts,

    [ValidatePattern("^[0-9A-Fa-f]{64}$")]
    [string] $ClipSha256,

    [string] $UnetCheckpoint,

    [string] $UnetArtifacts,

    [string] $UnetWeights,

    [ValidatePattern("^[0-9A-Fa-f]{64}$")]
    [string] $UnetSha256,

    [string] $VaeCheckpoint,

    [string] $VaeArtifacts,

    [string] $VaeWeights,

    [ValidatePattern("^[0-9A-Fa-f]{64}$")]
    [string] $VaeSha256
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param(
        [Parameter(Mandatory)]
        [string] $Path
    )

    if ([System.IO.Path]::IsPathFullyQualified($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine((Get-Location).ProviderPath, $Path)
    )
}

function Write-Utf8FileExclusive {
    param(
        [Parameter(Mandatory)]
        [string] $Path,

        [Parameter(Mandatory)]
        [AllowEmptyString()]
        [string] $Content
    )

    $encoding = [System.Text.UTF8Encoding]::new($false)
    $stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::Read
    )
    try {
        $writer = [System.IO.StreamWriter]::new($stream, $encoding)
        try {
            $writer.Write($Content)
            $writer.Flush()
        }
        finally {
            $writer.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Write-JsonFileExclusive {
    param(
        [Parameter(Mandatory)]
        [string] $Path,

        [Parameter(Mandatory)]
        [object] $Value
    )

    $parent = [System.IO.Path]::GetDirectoryName($Path)
    $temporary = [System.IO.Path]::Combine(
        $parent,
        ".$([System.IO.Path]::GetFileName($Path)).$([guid]::NewGuid().ToString('N')).tmp"
    )
    try {
        $json = $Value | ConvertTo-Json -Depth 100
        Write-Utf8FileExclusive -Path $temporary -Content ($json + [Environment]::NewLine)
        [System.IO.File]::Move($temporary, $Path, $false)
    }
    finally {
        if ([System.IO.File]::Exists($temporary)) {
            [System.IO.File]::Delete($temporary)
        }
    }
}

function Get-RelativeResultPath {
    param(
        [Parameter(Mandatory)]
        [string] $Path
    )

    return [System.IO.Path]::GetRelativePath($script:ResultsRoot, $Path).Replace("\", "/")
}

function Add-Argument {
    param(
        [Parameter(Mandatory)]
        [System.Collections.Generic.List[string]] $Arguments,

        [Parameter(Mandatory)]
        [string] $Name,

        [AllowNull()]
        [string] $Value
    )

    if (-not [string]::IsNullOrWhiteSpace($Value)) {
        [void] $Arguments.Add($Name)
        [void] $Arguments.Add($Value)
    }
}

function Get-CommonArguments {
    $arguments = [System.Collections.Generic.List[string]]::new()
    [void] $arguments.Add("--workers")
    [void] $arguments.Add($Workers.ToString([Globalization.CultureInfo]::InvariantCulture))
    [void] $arguments.Add("--delivery-queue-depth")
    [void] $arguments.Add($DeliveryQueueDepth.ToString([Globalization.CultureInfo]::InvariantCulture))
    [void] $arguments.Add("--dispatch-lookahead")
    [void] $arguments.Add($DispatchLookahead.ToString([Globalization.CultureInfo]::InvariantCulture))
    [void] $arguments.Add("--source-bytes")
    [void] $arguments.Add($SourceBytes.ToString([Globalization.CultureInfo]::InvariantCulture))
    [void] $arguments.Add("--scratch-bytes")
    [void] $arguments.Add($ScratchBytes.ToString([Globalization.CultureInfo]::InvariantCulture))
    [void] $arguments.Add("--prepared-bytes")
    [void] $arguments.Add($PreparedBytes.ToString([Globalization.CultureInfo]::InvariantCulture))
    [void] $arguments.Add("--consume")
    [void] $arguments.Add($Consumption)
    Add-Argument $arguments "--clip-checkpoint" $ClipCheckpoint
    Add-Argument $arguments "--clip-artifacts" $ClipArtifacts
    Add-Argument $arguments "--clip-sha256" $ClipSha256
    Add-Argument $arguments "--unet-checkpoint" $UnetCheckpoint
    Add-Argument $arguments "--unet-artifacts" $UnetArtifacts
    Add-Argument $arguments "--unet-weights" $UnetWeights
    Add-Argument $arguments "--unet-sha256" $UnetSha256
    Add-Argument $arguments "--vae-checkpoint" $VaeCheckpoint
    Add-Argument $arguments "--vae-artifacts" $VaeArtifacts
    Add-Argument $arguments "--vae-weights" $VaeWeights
    Add-Argument $arguments "--vae-sha256" $VaeSha256
    return $arguments.ToArray()
}

function Assert-ReportShape {
    param(
        [Parameter(Mandatory)]
        [object] $Report,

        [Parameter(Mandatory)]
        [string] $ExpectedCommand,

        [AllowNull()]
        [string] $ExpectedLane
    )

    if ($null -eq $Report.PSObject.Properties["schema_version"]) {
        throw "Benchmark report has no schema_version."
    }
    if ([string] $Report.command -ne $ExpectedCommand) {
        throw "Expected a '$ExpectedCommand' report, but the child emitted '$($Report.command)'."
    }
    $expectedConsumption = if ($ExpectedCommand -eq "validate") { "sha256" } else { $Consumption }
    if ([string] $Report.consumption -ne $expectedConsumption) {
        throw "Expected '$expectedConsumption' consumption, but the child emitted '$($Report.consumption)'."
    }
    if ($null -eq $Report.contract -or
        [string]::IsNullOrWhiteSpace([string] $Report.contract.digest_sha256)) {
        throw "Benchmark report has no contract.digest_sha256."
    }
    if ($ExpectedCommand -in @("sample", "prime")) {
        if ($null -eq $Report.lanes -or $Report.lanes.Count -ne 1) {
            throw "$ExpectedCommand report must contain exactly one lane payload."
        }
        $lane = $Report.lanes[0]
        $expected = if ($ExpectedCommand -eq "prime") { "model-weights" } else { $ExpectedLane }
        if ([string] $lane.lane -ne $expected) {
            throw "Expected lane '$expected', but the child emitted '$($lane.lane)'."
        }
    }
}

function Invoke-BenchmarkProcess {
    param(
        [Parameter(Mandatory)]
        [string[]] $Arguments,

        [Parameter(Mandatory)]
        [ValidatePattern("^[a-z0-9-]+$")]
        [string] $Label,

        [Parameter(Mandatory)]
        [string] $ExpectedCommand,

        [AllowNull()]
        [string] $ExpectedLane,

        [Parameter(Mandatory)]
        [bool] $Measured,

        [int] $Round = 0,

        [string] $Position = ""
    )

    $script:InvocationSequence += 1
    $prefix = "{0:D4}-{1}" -f $script:InvocationSequence, $Label
    $reportPath = [System.IO.Path]::Combine($script:RawDirectory, "$prefix.report.json")
    $stderrPath = [System.IO.Path]::Combine($script:RawDirectory, "$prefix.stderr.txt")
    $processPath = [System.IO.Path]::Combine($script:RawDirectory, "$prefix.process.json")

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $script:ExecutablePath
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.WorkingDirectory = $PSScriptRoot
    foreach ($argument in $Arguments) {
        [void] $startInfo.ArgumentList.Add($argument)
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $startedAt = [DateTimeOffset]::UtcNow
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $peakWorkingSet = [int64] 0
    $reportedPeakWorkingSet = [int64] 0
    $stdout = ""
    $stderr = ""
    $exitCode = $null
    $startError = $null

    try {
        try {
            if (-not $process.Start()) {
                throw "Process.Start returned false."
            }
            $stdoutTask = $process.StandardOutput.ReadToEndAsync()
            $stderrTask = $process.StandardError.ReadToEndAsync()

            while ($true) {
                try {
                    $process.Refresh()
                    $workingSet = [int64] $process.WorkingSet64
                    if ($workingSet -gt $peakWorkingSet) {
                        $peakWorkingSet = $workingSet
                    }
                }
                catch [System.InvalidOperationException] {
                    # The child may exit between HasExited and Refresh.
                }

                if ($process.HasExited) {
                    break
                }
                Start-Sleep -Milliseconds $PollIntervalMilliseconds
            }

            $process.WaitForExit()
            $stdout = $stdoutTask.GetAwaiter().GetResult()
            $stderr = $stderrTask.GetAwaiter().GetResult()
            $exitCode = $process.ExitCode
            try {
                $reportedPeakWorkingSet = [int64] $process.PeakWorkingSet64
            }
            catch [System.InvalidOperationException] {
                $reportedPeakWorkingSet = 0
            }
        }
        catch {
            $startError = $_.Exception.Message
        }
    }
    finally {
        $stopwatch.Stop()
        $process.Dispose()
    }

    Write-Utf8FileExclusive -Path $reportPath -Content $stdout
    Write-Utf8FileExclusive -Path $stderrPath -Content $stderr

    $report = $null
    $parseError = $null
    if (-not [string]::IsNullOrWhiteSpace($stdout)) {
        try {
            $report = $stdout | ConvertFrom-Json -Depth 100
        }
        catch {
            $parseError = $_.Exception.Message
        }
    }
    else {
        $parseError = "stdout was empty"
    }

    $completedAt = [DateTimeOffset]::UtcNow
    $processRecord = [ordered] @{
        schema_version = 1
        invocation = $script:InvocationSequence
        label = $Label
        measured = $Measured
        round = $Round
        position = $Position
        expected_command = $ExpectedCommand
        expected_lane = $ExpectedLane
        executable = $script:ExecutablePath
        arguments = $Arguments
        started_at_utc = $startedAt.ToString("O")
        completed_at_utc = $completedAt.ToString("O")
        wall_clock_ms = $stopwatch.Elapsed.TotalMilliseconds
        poll_interval_ms = $PollIntervalMilliseconds
        peak_working_set64_polled_bytes = $peakWorkingSet
        peak_working_set64_reported_bytes = $reportedPeakWorkingSet
        exit_code = $exitCode
        start_error = $startError
        json_parse_error = $parseError
        report_path = Get-RelativeResultPath $reportPath
        stderr_path = Get-RelativeResultPath $stderrPath
    }
    Write-JsonFileExclusive -Path $processPath -Value $processRecord

    if ($null -ne $startError) {
        throw "Failed to execute '$script:ExecutablePath': $startError. Raw process record: $processPath"
    }
    if ($exitCode -ne 0) {
        throw "Benchmark child '$Label' exited with code $exitCode. Stderr: $stderrPath"
    }
    if ($null -ne $parseError) {
        throw "Benchmark child '$Label' did not emit valid JSON: $parseError. Report: $reportPath"
    }
    Assert-ReportShape -Report $report -ExpectedCommand $ExpectedCommand -ExpectedLane $ExpectedLane

    return [pscustomobject] @{
        Report = $report
        Process = [pscustomobject] $processRecord
        ReportPath = Get-RelativeResultPath $reportPath
        StderrPath = Get-RelativeResultPath $stderrPath
        ProcessPath = Get-RelativeResultPath $processPath
    }
}

function Invoke-Prime {
    param(
        [Parameter(Mandatory)]
        [string] $Label,

        [int] $Round = 0,

        [string] $Position = ""
    )

    $arguments = [System.Collections.Generic.List[string]]::new()
    [void] $arguments.Add("prime")
    [void] $arguments.Add("--cache")
    [void] $arguments.Add($script:CachePath)
    foreach ($argument in $script:CommonArguments) {
        [void] $arguments.Add($argument)
    }
    return Invoke-BenchmarkProcess `
        -Arguments $arguments.ToArray() `
        -Label $Label `
        -ExpectedCommand "prime" `
        -ExpectedLane $null `
        -Measured $false `
        -Round $Round `
        -Position $Position
}

function Invoke-Lane {
    param(
        [Parameter(Mandatory)]
        [ValidateSet("legacy", "model-weights")]
        [string] $Lane,

        [Parameter(Mandatory)]
        [string] $Phase,

        [Parameter(Mandatory)]
        [bool] $Measured,

        [int] $Round = 0,

        [string] $Position = ""
    )

    $primeResult = $null
    if ($Lane -eq "model-weights" -and $PrimePrepared) {
        $primeResult = Invoke-Prime `
            -Label "$Phase-model-weights-prime" `
            -Round $Round `
            -Position $Position
    }

    $arguments = [System.Collections.Generic.List[string]]::new()
    [void] $arguments.Add("sample")
    [void] $arguments.Add("--lane")
    [void] $arguments.Add($Lane)
    if ($Lane -eq "model-weights" -and $null -ne $script:CachePath) {
        [void] $arguments.Add("--cache")
        [void] $arguments.Add($script:CachePath)
        if ($ResetPrepared) {
            [void] $arguments.Add("--reset-prepared")
        }
    }
    foreach ($argument in $script:CommonArguments) {
        [void] $arguments.Add($argument)
    }

    $sampleResult = Invoke-BenchmarkProcess `
        -Arguments $arguments.ToArray() `
        -Label "$Phase-$Lane" `
        -ExpectedCommand "sample" `
        -ExpectedLane $Lane `
        -Measured $Measured `
        -Round $Round `
        -Position $Position

    return [pscustomobject] @{
        Sample = $sampleResult
        Prime = $primeResult
    }
}

function Add-Number {
    param(
        [Parameter(Mandatory)]
        [System.Collections.Specialized.OrderedDictionary] $Metrics,

        [Parameter(Mandatory)]
        [string] $Name,

        [AllowNull()]
        [object] $Value
    )

    if ($null -eq $Value) {
        return
    }
    $number = 0.0
    if (-not [double]::TryParse(
        [string] $Value,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref] $number
    )) {
        return
    }
    if (-not $Metrics.Contains($Name)) {
        $Metrics[$Name] = [System.Collections.Generic.List[double]]::new()
    }
    [void] $Metrics[$Name].Add($number)
}

function Add-TimedProperties {
    param(
        [Parameter(Mandatory)]
        [System.Collections.Specialized.OrderedDictionary] $Metrics,

        [AllowNull()]
        [object] $Value,

        [string] $Path = ""
    )

    if ($null -eq $Value -or $Value -is [string] -or $Value -is [System.Array]) {
        return
    }
    foreach ($property in $Value.PSObject.Properties) {
        $propertyPath = if ([string]::IsNullOrEmpty($Path)) {
            $property.Name
        }
        else {
            "$Path.$($property.Name)"
        }
        if ($property.Name -match "(?:_ms|milliseconds)$") {
            Add-Number -Metrics $Metrics -Name $propertyPath -Value $property.Value
        }
        elseif ($null -ne $property.Value -and
            $property.Value -isnot [string] -and
            $property.Value -isnot [ValueType] -and
            $property.Value -isnot [System.Array]) {
            Add-TimedProperties -Metrics $Metrics -Value $property.Value -Path $propertyPath
        }
    }
}

function Get-Statistics {
    param(
        [Parameter(Mandatory)]
        [double[]] $Values
    )

    if ($Values.Count -eq 0) {
        throw "Cannot summarize an empty metric."
    }
    $sorted = @($Values | Sort-Object)
    $count = $sorted.Count
    if (($count % 2) -eq 1) {
        $median = [double] $sorted[[int] [Math]::Floor($count / 2)]
    }
    else {
        $upper = [int] ($count / 2)
        $median = ([double] $sorted[$upper - 1] + [double] $sorted[$upper]) / 2.0
    }
    $p95Index = [Math]::Max(0, [int] [Math]::Ceiling(0.95 * $count) - 1)
    $sum = 0.0
    foreach ($value in $sorted) {
        $sum += [double] $value
    }
    return [ordered] @{
        count = $count
        min = [double] $sorted[0]
        median = $median
        p95_nearest_rank = [double] $sorted[$p95Index]
        max = [double] $sorted[$count - 1]
        mean = $sum / $count
    }
}

function Get-MetricSummaries {
    param(
        [Parameter(Mandatory)]
        [System.Collections.Specialized.OrderedDictionary] $Metrics
    )

    $summaries = [ordered] @{}
    foreach ($name in @($Metrics.Keys | Sort-Object)) {
        $summaries[$name] = Get-Statistics -Values $Metrics[$name].ToArray()
    }
    return $summaries
}

function Convert-InvocationToRecord {
    param(
        [Parameter(Mandatory)]
        [object] $Result,

        [Parameter(Mandatory)]
        [string] $PairOrder
    )

    $lane = $Result.Report.lanes[0]
    $outputSetSha256 = if ($null -ne $lane.PSObject.Properties["output_set_sha256"]) {
        $lane.output_set_sha256
    }
    else {
        $null
    }

    return [ordered] @{
        round = $Result.Process.round
        position = $Result.Process.position
        pair_order = $PairOrder
        lane = $lane.lane
        report_schema_version = $Result.Report.schema_version
        contract_digest_sha256 = $Result.Report.contract.digest_sha256
        output_set_sha256 = $outputSetSha256
        wall_clock_ms = $Result.Process.wall_clock_ms
        peak_working_set64_polled_bytes = $Result.Process.peak_working_set64_polled_bytes
        peak_working_set64_reported_bytes = $Result.Process.peak_working_set64_reported_bytes
        report_path = $Result.ReportPath
        stderr_path = $Result.StderrPath
        process_path = $Result.ProcessPath
    }
}

if ($ResetPrepared -and $PrimePrepared) {
    throw "-ResetPrepared and -PrimePrepared are mutually exclusive."
}
if (($ResetPrepared -or $PrimePrepared) -and
    [string]::IsNullOrWhiteSpace($CacheDirectory)) {
    throw "-ResetPrepared and -PrimePrepared require -CacheDirectory."
}

$script:ResultsRoot = Resolve-AbsolutePath $ResultsDirectory
if ([System.IO.File]::Exists($script:ResultsRoot)) {
    throw "ResultsDirectory points to a file: $script:ResultsRoot"
}
[void] [System.IO.Directory]::CreateDirectory($script:ResultsRoot)

$aggregatePath = [System.IO.Path]::Combine($script:ResultsRoot, "aggregate.json")
if ([System.IO.File]::Exists($aggregatePath)) {
    throw "Refusing to overwrite existing aggregate: $aggregatePath"
}

$script:RawDirectory = [System.IO.Path]::Combine($script:ResultsRoot, "raw")
if ([System.IO.File]::Exists($script:RawDirectory)) {
    throw "Raw output path points to a file: $script:RawDirectory"
}
if ([System.IO.Directory]::Exists($script:RawDirectory) -and
    [System.IO.Directory]::EnumerateFileSystemEntries($script:RawDirectory).GetEnumerator().MoveNext()) {
    throw "Refusing to mix results with non-empty raw directory: $script:RawDirectory"
}
[void] [System.IO.Directory]::CreateDirectory($script:RawDirectory)

if ([string]::IsNullOrWhiteSpace($Executable)) {
    $suffix = if ([OperatingSystem]::IsWindows()) { ".exe" } else { "" }
    $Executable = [System.IO.Path]::Combine(
        $PSScriptRoot,
        "target",
        "release",
        "dinoml-sd15-loader-benchmark$suffix"
    )
}
$script:ExecutablePath = Resolve-AbsolutePath $Executable
if (-not [System.IO.File]::Exists($script:ExecutablePath)) {
    throw "Release benchmark executable not found: $script:ExecutablePath"
}
$executableInfo = [System.IO.FileInfo]::new($script:ExecutablePath)
$executableSha256 = (Get-FileHash -LiteralPath $script:ExecutablePath -Algorithm SHA256).Hash.ToLowerInvariant()

$script:CachePath = $null
if (-not [string]::IsNullOrWhiteSpace($CacheDirectory)) {
    $script:CachePath = Resolve-AbsolutePath $CacheDirectory
    if ([System.IO.File]::Exists($script:CachePath)) {
        throw "CacheDirectory points to a file: $script:CachePath"
    }
    [void] [System.IO.Directory]::CreateDirectory($script:CachePath)
}

$preparedCacheState = if ($null -eq $script:CachePath) {
    "disabled"
}
elseif ($ResetPrepared) {
    "reset-per-model-weights-sample"
}
elseif ($PrimePrepared) {
    "prime-in-fresh-process-before-model-weights-sample"
}
else {
    "reuse"
}

$script:CommonArguments = @(Get-CommonArguments)
$script:InvocationSequence = 0
$startedRun = [DateTimeOffset]::UtcNow
$validationResult = $null
$setupRecords = [System.Collections.Generic.List[object]]::new()
$primeRecords = [System.Collections.Generic.List[object]]::new()
$measuredRecords = [System.Collections.Generic.List[object]]::new()
$laneResults = [ordered] @{
    legacy = [System.Collections.Generic.List[object]]::new()
    "model-weights" = [System.Collections.Generic.List[object]]::new()
}
$canonicalContractDigest = $null
$canonicalContract = $null
$canonicalComponentIdentities = $null
$canonicalOutputSetSha256 = $null

function Get-ContractComponentIdentities {
    param(
        [Parameter(Mandatory)]
        [object] $Contract
    )

    if ($null -eq $Contract.PSObject.Properties["components"]) {
        throw "Benchmark contract has no components."
    }

    $components = @($Contract.components)
    if ($components.Count -eq 0) {
        throw "Benchmark contract has no component identities."
    }

    $identities = [System.Collections.Generic.List[object]]::new()
    foreach ($component in $components) {
        foreach ($propertyName in @("component", "checkpoint_sha256", "identity_source")) {
            if ($null -eq $component.PSObject.Properties[$propertyName] -or
                [string]::IsNullOrWhiteSpace([string] $component.$propertyName)) {
                throw "Benchmark contract component has no '$propertyName'."
            }
        }
        [void] $identities.Add([pscustomobject] [ordered] @{
            component = [string] $component.component
            checkpoint_sha256 = [string] $component.checkpoint_sha256
            identity_source = [string] $component.identity_source
        })
    }

    return $identities.ToArray()
}

function Assert-ConsistentContract {
    param(
        [Parameter(Mandatory)]
        [object] $Result
    )

    $digest = [string] $Result.Report.contract.digest_sha256
    $componentIdentities = @(Get-ContractComponentIdentities $Result.Report.contract)
    if ($null -eq $script:canonicalContractDigest) {
        $script:canonicalContractDigest = $digest
        $script:canonicalContract = $Result.Report.contract
        $script:canonicalComponentIdentities = $componentIdentities
    }
    elseif ($digest -ne $script:canonicalContractDigest) {
        throw "Contract digest changed from '$script:canonicalContractDigest' to '$digest'."
    }
    if ($componentIdentities.Count -ne $script:canonicalComponentIdentities.Count) {
        throw "Contract component identity count changed from '$($script:canonicalComponentIdentities.Count)' to '$($componentIdentities.Count)'."
    }
    for ($index = 0; $index -lt $componentIdentities.Count; $index += 1) {
        $expected = $script:canonicalComponentIdentities[$index]
        $actual = $componentIdentities[$index]
        if ($actual.component -ne $expected.component -or
            $actual.checkpoint_sha256 -ne $expected.checkpoint_sha256 -or
            $actual.identity_source -ne $expected.identity_source) {
            throw "Contract component identity at index $index changed from '$($expected.component):$($expected.checkpoint_sha256):$($expected.identity_source)' to '$($actual.component):$($actual.checkpoint_sha256):$($actual.identity_source)'."
        }
    }

    if ([string] $Result.Report.consumption -eq "sha256") {
        foreach ($lane in @($Result.Report.lanes)) {
            $outputDigest = [string] $lane.output_set_sha256
            if ([string]::IsNullOrWhiteSpace($outputDigest)) {
                throw "SHA-256 consumption report for lane '$($lane.lane)' has no output_set_sha256."
            }
            if ($null -eq $script:canonicalOutputSetSha256) {
                $script:canonicalOutputSetSha256 = $outputDigest
            }
            elseif ($outputDigest -ne $script:canonicalOutputSetSha256) {
                throw "Output set digest changed from '$script:canonicalOutputSetSha256' to '$outputDigest' for lane '$($lane.lane)'."
            }
        }
    }
}

Write-Host "Benchmark executable: $script:ExecutablePath"
Write-Host "Results directory: $script:ResultsRoot"
Write-Host "Samples per lane: $Samples; consumption: $Consumption; prepared cache state: $preparedCacheState"
Write-Host "Execution: workers=$Workers; result queue=$DeliveryQueueDepth; lookahead=$DispatchLookahead"
Write-Host "Budgets: source=$SourceBytes; scratch=$ScratchBytes; prepared=$PreparedBytes bytes"

if (-not $SkipValidation) {
    Write-Host "Validating lane equivalence..."
    $validationArguments = [System.Collections.Generic.List[string]]::new()
    [void] $validationArguments.Add("validate")
    foreach ($argument in $script:CommonArguments) {
        [void] $validationArguments.Add($argument)
    }
    $validationResult = Invoke-BenchmarkProcess `
        -Arguments $validationArguments.ToArray() `
        -Label "validation" `
        -ExpectedCommand "validate" `
        -ExpectedLane $null `
        -Measured $false
    Assert-ConsistentContract $validationResult
    if ($null -eq $validationResult.Report.validation -or
        -not [bool] $validationResult.Report.validation.matched) {
        throw "Legacy and model-weights validation did not match. Report: $($validationResult.ReportPath)"
    }
}

Write-Host "Running and discarding one setup pair (legacy/model-weights)..."
$setupLegacy = Invoke-Lane `
    -Lane "legacy" `
    -Phase "setup-a" `
    -Measured $false `
    -Round 0 `
    -Position "A"
Assert-ConsistentContract $setupLegacy.Sample
[void] $setupRecords.Add((Convert-InvocationToRecord $setupLegacy.Sample "AB"))
if ($null -ne $setupLegacy.Prime) {
    [void] $primeRecords.Add($setupLegacy.Prime.Process)
}

$setupModel = Invoke-Lane `
    -Lane "model-weights" `
    -Phase "setup-b" `
    -Measured $false `
    -Round 0 `
    -Position "B"
Assert-ConsistentContract $setupModel.Sample
[void] $setupRecords.Add((Convert-InvocationToRecord $setupModel.Sample "AB"))
if ($null -ne $setupModel.Prime) {
    [void] $primeRecords.Add($setupModel.Prime.Process)
}

for ($round = 1; $round -le $Samples; $round += 1) {
    $pairOrder = if (($round % 2) -eq 1) { "AB" } else { "BA" }
    $order = if ($pairOrder -eq "AB") {
        @("legacy", "model-weights")
    }
    else {
        @("model-weights", "legacy")
    }
    Write-Host "Measured pair $round/$Samples ($pairOrder)..."
    for ($positionIndex = 0; $positionIndex -lt $order.Count; $positionIndex += 1) {
        $lane = $order[$positionIndex]
        $position = if ($positionIndex -eq 0) { "A" } else { "B" }
        $phase = "sample-{0:D3}-{1}" -f $round, $position.ToLowerInvariant()
        $laneRun = Invoke-Lane `
            -Lane $lane `
            -Phase $phase `
            -Measured $true `
            -Round $round `
            -Position $position
        Assert-ConsistentContract $laneRun.Sample
        [void] $laneResults[$lane].Add($laneRun.Sample)
        [void] $measuredRecords.Add((Convert-InvocationToRecord $laneRun.Sample $pairOrder))
        if ($null -ne $laneRun.Prime) {
            [void] $primeRecords.Add($laneRun.Prime.Process)
        }
    }
}

$laneSummaries = [ordered] @{}
foreach ($lane in @("legacy", "model-weights")) {
    $timedMetrics = [ordered] @{}
    $peakMetrics = [ordered] @{}
    $throughputMetrics = [ordered] @{}
    foreach ($result in $laneResults[$lane]) {
        Add-TimedProperties -Metrics $timedMetrics -Value $result.Report
        Add-TimedProperties -Metrics $timedMetrics -Value $result.Report.lanes[0] -Path "lane"
        Add-Number `
            -Metrics $timedMetrics `
            -Name "process.wall_clock_ms" `
            -Value $result.Process.wall_clock_ms
        Add-Number `
            -Metrics $peakMetrics `
            -Name "peak_working_set64_polled_bytes" `
            -Value $result.Process.peak_working_set64_polled_bytes
        Add-Number `
            -Metrics $peakMetrics `
            -Name "peak_working_set64_reported_bytes" `
            -Value $result.Process.peak_working_set64_reported_bytes
        if ($null -ne $result.Report.lanes[0].PSObject.Properties["throughput_mib_per_second"]) {
            Add-Number `
                -Metrics $throughputMetrics `
                -Name "throughput_mib_per_second" `
                -Value $result.Report.lanes[0].throughput_mib_per_second
        }
    }
    $laneSummaries[$lane] = [ordered] @{
        sample_count = $laneResults[$lane].Count
        timed_fields = Get-MetricSummaries $timedMetrics
        memory_fields = Get-MetricSummaries $peakMetrics
        rate_fields = Get-MetricSummaries $throughputMetrics
    }
}

$validationRecord = if ($null -eq $validationResult) {
    [ordered] @{
        skipped = $true
    }
}
else {
    [ordered] @{
        skipped = $false
        matched = [bool] $validationResult.Report.validation.matched
        target_count = $validationResult.Report.validation.target_count
        target_bytes = $validationResult.Report.validation.target_bytes
        legacy_set_sha256 = $validationResult.Report.validation.legacy_set_sha256
        model_weights_set_sha256 = $validationResult.Report.validation.model_weights_set_sha256
        report_path = $validationResult.ReportPath
        stderr_path = $validationResult.StderrPath
        process_path = $validationResult.ProcessPath
    }
}

$completedRun = [DateTimeOffset]::UtcNow
$aggregate = [ordered] @{
    schema_version = 2
    kind = "dinoml-sd15-loader-samples"
    generated_at_utc = $completedRun.ToString("O")
    run = [ordered] @{
        started_at_utc = $startedRun.ToString("O")
        completed_at_utc = $completedRun.ToString("O")
        elapsed_ms = ($completedRun - $startedRun).TotalMilliseconds
        results_directory = $script:ResultsRoot
        raw_directory = Get-RelativeResultPath $script:RawDirectory
    }
    protocol = [ordered] @{
        build_profile = "release"
        fresh_process_per_sample = $true
        discarded_setup_pairs = 1
        measured_pairs = $Samples
        samples_per_lane = $Samples
        pair_order = "odd rounds AB (legacy/model-weights), even rounds BA (model-weights/legacy)"
        percentile = "nearest-rank p95 (ceil(0.95 * n))"
        median = "middle value for odd n; arithmetic mean of two middle values for even n"
        filesystem_cache_state = "uncontrolled"
        consumption = $Consumption
        working_set_measurement = "WorkingSet64 polled from the child process"
        poll_interval_ms = $PollIntervalMilliseconds
    }
    executable = [ordered] @{
        path = $script:ExecutablePath
        sha256 = $executableSha256
        bytes = $executableInfo.Length
        last_write_time_utc = $executableInfo.LastWriteTimeUtc.ToString("O")
    }
    host = [ordered] @{
        os_description = [Runtime.InteropServices.RuntimeInformation]::OSDescription
        os_architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
        process_architecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
        machine_name = [Environment]::MachineName
        processor_count = [Environment]::ProcessorCount
        powershell_version = $PSVersionTable.PSVersion.ToString()
    }
    configuration = [ordered] @{
        workers = $Workers
        delivery_queue_depth = $DeliveryQueueDepth
        dispatch_lookahead = $DispatchLookahead
        source_bytes = $SourceBytes
        scratch_bytes = $ScratchBytes
        prepared_bytes = $PreparedBytes
        consumption = $Consumption
        prepared_cache_state = $preparedCacheState
        cache_directory = $script:CachePath
        reset_prepared = [bool] $ResetPrepared
        prime_prepared = [bool] $PrimePrepared
        skip_validation = [bool] $SkipValidation
        path_overrides = [ordered] @{
            clip_checkpoint = $ClipCheckpoint
            clip_artifacts = $ClipArtifacts
            unet_checkpoint = $UnetCheckpoint
            unet_artifacts = $UnetArtifacts
            unet_weights = $UnetWeights
            vae_checkpoint = $VaeCheckpoint
            vae_artifacts = $VaeArtifacts
            vae_weights = $VaeWeights
        }
        trusted_checkpoint_sha256 = [ordered] @{
            clip = $ClipSha256
            unet = $UnetSha256
            vae = $VaeSha256
        }
    }
    contract = $canonicalContract
    output_set_sha256 = $canonicalOutputSetSha256
    validation = $validationRecord
    lanes = $laneSummaries
    measured_samples = $measuredRecords
    discarded_setup_samples = $setupRecords
    prime_invocations = $primeRecords
}

Write-JsonFileExclusive -Path $aggregatePath -Value $aggregate
Write-Host "Wrote aggregate benchmark report: $aggregatePath"
$aggregatePath
