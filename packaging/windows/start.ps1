[CmdletBinding()]
param([string]$SourceRoot, [switch]$Rebuild)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Invoke-Docker {
    param([string[]]$DockerArguments)
    & docker @DockerArguments
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed (exit $LASTEXITCODE)." }
}

function Test-ReleaseImage {
    param($Release)
    try {
        $imageJSON = & docker image inspect $Release.image 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        $details = ($imageJSON | ConvertFrom-Json)[0]
        return ($details.Os -eq 'linux' -and $details.Architecture -eq 'amd64' -and
            ($details.RootFS.Layers -join ',') -ceq ($Release.rootfs_layers -join ','))
    }
    catch { return $false }
}

function Get-PartHash {
    param([string]$Path)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try { return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}

function Import-ReleaseImage {
    param($Release, [string]$Root)
    $archiveHash = [System.Security.Cryptography.SHA256]::Create()
    $buffer = New-Object byte[] (1024 * 1024)
    try {
        foreach ($part in $Release.parts) {
            if ($part.name -notmatch '^imdb-sentiment-api-[a-zA-Z0-9._-]+\.part\d{3}$' -or
                $part.sha256 -notmatch '^[0-9a-f]{64}$' -or $part.size -le 0) {
                throw 'Invalid archive part in release manifest.'
            }
            $partPath = Join-Path $Root $part.name
            $valid = (Test-Path -LiteralPath $partPath -PathType Leaf)
            if ($valid) { $valid = ((Get-Item -LiteralPath $partPath).Length -eq $part.size -and (Get-PartHash $partPath) -eq $part.sha256) }
            if (-not $valid) {
                $expectedPrefix = "https://github.com/RetajMelhem/imdb-sentiment-api/releases/download/v$($Release.version)/"
                if ($part.url -ne ($expectedPrefix + $part.name)) { throw 'Invalid release download URL.' }
                Write-Host "Downloading $($part.name)..."
                [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
                $partialPath = $partPath + '.download'
                for ($attempt = 1; $attempt -le 3; $attempt++) {
                    try {
                        Invoke-WebRequest -Uri $part.url -OutFile $partialPath -UseBasicParsing
                        if ((Get-Item -LiteralPath $partialPath).Length -ne $part.size -or (Get-PartHash $partialPath) -ne $part.sha256) {
                            throw "Checksum mismatch for $($part.name)."
                        }
                        break
                    }
                    catch {
                        if ($attempt -eq 3) { throw }
                        Write-Host "Retrying $($part.name)..."
                        Start-Sleep -Seconds (2 * $attempt)
                    }
                }
                [IO.File]::Delete($partPath)
                [IO.File]::Move($partialPath, $partPath)
            }
            $stream = [IO.File]::OpenRead($partPath)
            try {
                while (($count = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    $null = $archiveHash.TransformBlock($buffer, 0, $count, $buffer, 0)
                }
            }
            finally { $stream.Dispose() }
        }
        $null = $archiveHash.TransformFinalBlock((New-Object byte[] 0), 0, 0)
        $actualHash = ([BitConverter]::ToString($archiveHash.Hash)).Replace('-', '').ToLowerInvariant()
        if ($actualHash -ne $Release.archive_sha256) { throw 'Complete image archive checksum mismatch.' }
    }
    finally { $archiveHash.Dispose() }

    Write-Host 'Importing the complete model and CUDA/CPU image into Docker...'
    $processInfo = New-Object Diagnostics.ProcessStartInfo
    $processInfo.FileName = (Get-Command docker).Source
    $processInfo.Arguments = 'image load'
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    $processInfo.RedirectStandardInput = $true
    $process = [Diagnostics.Process]::Start($processInfo)
    try {
        foreach ($part in $Release.parts) {
            $stream = [IO.File]::OpenRead((Join-Path $Root $part.name))
            try { $stream.CopyTo($process.StandardInput.BaseStream) }
            finally { $stream.Dispose() }
        }
        $process.StandardInput.Close()
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) { throw 'Docker could not import the release image.' }
    }
    finally { $process.Dispose() }
    if (-not (Test-ReleaseImage $Release)) { throw 'Loaded image differs from the release.' }
}

try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Install and start Docker Desktop first.' }
    $engineOS = & docker info --format '{{.OSType}}' 2>$null
    if ($LASTEXITCODE -ne 0 -or $engineOS -ne 'linux') { throw 'Start Docker Desktop using Linux containers.' }
    Invoke-Docker -DockerArguments @('compose', 'version')
    if ($SourceRoot) {
        $root = (Resolve-Path -LiteralPath $SourceRoot).Path
        $composePath = Join-Path $root 'docker-compose.yml'
        $gpuPath = Join-Path $root 'docker-compose.gpu.yml'
        $composeArgs = @('compose', '--project-directory', $root, '--file', $composePath)
        $imageName = (& docker @composeArgs config --images | Select-Object -First 1)
        $existingImage = & docker image ls --quiet $imageName
        if ($Rebuild -or -not $existingImage) {
            Invoke-Docker -DockerArguments ($composeArgs + @('build'))
        }
    }
    else {
        $root = $PSScriptRoot
        $composePath = Join-Path $root 'compose.yaml'
        $gpuPath = Join-Path $root 'compose.gpu.yaml'
        $release = Get-Content -LiteralPath (Join-Path $root 'release.json') -Raw | ConvertFrom-Json
        if ($release.schema_version -ne 2 -or $release.image -notmatch '^imdb-sentiment-api:\d+\.\d+\.\d+$' -or
            $release.version -notmatch '^\d+\.\d+\.\d+$' -or $release.archive_sha256 -notmatch '^[0-9a-f]{64}$' -or
            -not $release.rootfs_layers -or -not $release.parts) { throw 'Invalid release manifest.' }
        if (-not (Test-ReleaseImage $release)) { Import-ReleaseImage $release $root }
        $imageName = $release.image
        $composeArgs = @('compose', '--project-directory', $root, '--file', $composePath)
    }

    $gpuAvailable = $false
    if ($env:DEVICE -ne 'cpu') {
        Write-Host 'Checking GPU access through Docker...'
        try {
            $probeInfo = New-Object Diagnostics.ProcessStartInfo
            $probeInfo.FileName = (Get-Command docker).Source
            $probeCode = "import torch; assert torch.cuda.is_available(); torch.ones(1, device='cuda'); print(torch.cuda.get_device_name(0))"
            $probeInfo.Arguments = 'run --rm --pull never --gpus all --entrypoint python ' + $imageName + ' -c "' + $probeCode + '"'
            $probeInfo.UseShellExecute = $false
            $probeInfo.CreateNoWindow = $true
            $probeInfo.RedirectStandardOutput = $true
            $probeInfo.RedirectStandardError = $true
            $probe = [Diagnostics.Process]::Start($probeInfo)
            try {
                $probeOutput = $probe.StandardOutput.ReadToEndAsync()
                $probeErrors = $probe.StandardError.ReadToEndAsync()
                $probe.WaitForExit()
                $gpuAvailable = ($probe.ExitCode -eq 0)
                if ($gpuAvailable) { Write-Host ('GPU available: ' + $probeOutput.Result.Trim()) }
            }
            finally { $probe.Dispose() }
        }
        catch { $gpuAvailable = $false }
    }
    if ($gpuAvailable) { $composeArgs += @('--file', $gpuPath) }
    else { Write-Host 'GPU unavailable or CPU requested; using CPU and system RAM.' }
    Invoke-Docker -DockerArguments ($composeArgs + @('up', '--no-build', '--pull', 'never', '-d', '--wait', '--wait-timeout', '240'))
    Invoke-Docker -DockerArguments ($composeArgs + @('exec', '-T', 'api', 'python', '-c', "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"))
    Write-Host 'Ready. Open http://localhost:8080/docs (or the PORT set in .env).'
    exit 0
}
catch {
    Write-Host "Startup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
