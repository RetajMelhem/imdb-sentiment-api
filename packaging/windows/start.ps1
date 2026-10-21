[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

function Invoke-Docker {
    param([string[]]$DockerArguments)
    & docker @DockerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed (exit $LASTEXITCODE)."
    }
}

function Test-ReleaseImage {
    param($Release)
    try {
        $imageJSON = & docker image inspect $Release.image 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        $imageDetails = ($imageJSON | ConvertFrom-Json)[0]
        # Layer identities survive OCI/classic Docker save/load conversions.
        return ($imageDetails.Os -eq 'linux' -and
            $imageDetails.Architecture -eq 'amd64' -and
            ($imageDetails.RootFS.Layers -join ',') -ceq ($Release.rootfs_layers -join ','))
    }
    catch { return $false }
}

try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Install and start Docker Desktop with Linux containers first.'
    }
    $engineOS = & docker info --format '{{.OSType}}' 2>$null
    if ($LASTEXITCODE -ne 0 -or $engineOS -ne 'linux') {
        throw 'Start Docker Desktop and switch to Linux containers, then retry.'
    }
    Invoke-Docker -DockerArguments @('compose', 'version')

    $releasePath = Join-Path $PSScriptRoot 'release.json'
    $release = Get-Content -LiteralPath $releasePath -Raw | ConvertFrom-Json
    if ($release.image -notmatch '^imdb-sentiment-api:\d+\.\d+\.\d+$' -or
        $release.image_id -notmatch '^sha256:[0-9a-f]{64}$' -or
        $release.archive_sha256 -notmatch '^[0-9a-f]{64}$' -or
        $release.archive -notmatch '^[a-zA-Z0-9._-]+\.tar$') {
        throw 'Invalid release manifest.'
    }
    if (-not $release.rootfs_layers -or @($release.rootfs_layers).Count -lt 1) {
        throw 'Missing image layer identities in release manifest.'
    }
    if (-not (Test-ReleaseImage -Release $release)) {
        $archivePath = Join-Path $PSScriptRoot $release.archive
        if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
            throw "Missing image archive: $($release.archive). Extract the complete ZIP."
        }
        Write-Host 'Verifying the bundled image archive...'
        $hashAlgorithm = [System.Security.Cryptography.SHA256]::Create()
        $archiveStream = [System.IO.File]::OpenRead($archivePath)
        try {
            $actualHash = ([System.BitConverter]::ToString(
                $hashAlgorithm.ComputeHash($archiveStream)
            )).Replace('-', '').ToLowerInvariant()
        }
        finally {
            $archiveStream.Dispose()
            $hashAlgorithm.Dispose()
        }
        if ($actualHash -ne $release.archive_sha256) {
            throw 'Image archive checksum mismatch. Download and extract the ZIP again.'
        }
        Write-Host 'Importing the model and API image into Docker...'
        Invoke-Docker -DockerArguments @('image', 'load', '--input', $archivePath)
        if (-not (Test-ReleaseImage -Release $release)) {
            throw 'Loaded image does not match this release.'
        }
    }

    Invoke-Docker -DockerArguments @(
        'compose', '--project-directory', $PSScriptRoot,
        '--file', (Join-Path $PSScriptRoot 'compose.yaml'),
        'up', '--no-build', '--pull', 'never', '-d', '--wait', '--wait-timeout', '180'
    )
    Write-Host 'Ready. Open http://localhost:8080/docs (or the PORT set in .env).'
    exit 0
}
catch {
    Write-Host "Startup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
