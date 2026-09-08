# Convert a WAV through the same pipeline the live path uses, for listening
# tests. No audio hardware involved.
#
#   .\convert.ps1 take3.wav out.wav
#   .\convert.ps1 take3.wav A.wav -Extra 5000
#   .\convert.ps1 take3.wav B.wav -Block 250
#   .\convert.ps1 take3.wav C.wav -Key 9 -Formant 1.5 -IndexRate 0
#
# ASCII only -- see the note in go-live.ps1. PowerShell 5.1 reads .ps1 as
# cp932 and mangles UTF-8 Japanese badly enough to break the parser.
#
# NOTE: this is NOT RVC's own file conversion. It replays the file through
# the realtime path (infer/rtrvc.py, block by block), which is the point when
# you are tuning the realtime settings -- and a trap when you are judging the
# model. RVC's WebUI "model inference" tab uses a different code path that
# sees the whole utterance at once, and it sounds better. Compare against it,
# do not mistake this for it.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$In,
    [Parameter(Mandatory = $true, Position = 1)][string]$Out,

    [int]$Key         = 12,
    [double]$Formant  = 0,
    [double]$IndexRate = 0.75,

    # Window. Leave at 0 to take the engine defaults (130 / 50 / 2500 for rvc).
    [double]$Block     = 0,
    [double]$Crossfade = 0,
    [double]$Extra     = 0,

    [string]$ModelGlob = "D:\Claude\Project\RVC\assets\weights\ena_e150_s*.pth",
    [string]$IndexGlob = "D:\Claude\Project\RVC\assets\indices\*ena*.index",
    [string]$Venv      = "D:\Claude\Project\.venv-rvc"
)

$ErrorActionPreference = "Stop"

if (-not $env:VIRTUAL_ENV) {
    $activate = Join-Path $Venv "Scripts\Activate.ps1"
    if (-not (Test-Path $activate)) {
        throw "venv not found: $activate  (pass -Venv if it moved)"
    }
    & $activate
}

Set-Location $PSScriptRoot

$model = (Get-ChildItem $ModelGlob -ErrorAction SilentlyContinue |
          Sort-Object Name | Select-Object -Last 1).FullName
if (-not $model) { throw "no speaker model matched: $ModelGlob" }

$cmd = @(
    "-m", "rtvc.realtime", "--engine", "rvc", "--offline",
    "--offline-input", $In,
    "--offline-out", $Out,
    "--rvc-model", $model,
    "--rvc-key", $Key,
    "--rvc-formant", $Formant
)

if ($IndexRate -gt 0) {
    $index = (Get-ChildItem $IndexGlob -ErrorAction SilentlyContinue |
              Sort-Object Name | Select-Object -Last 1).FullName
    if (-not $index) { throw "index rate is $IndexRate but no index matched: $IndexGlob" }
    $cmd += @("--rvc-index", $index, "--rvc-index-rate", $IndexRate)
}

if ($Block -gt 0)     { $cmd += @("--block-ms", $Block) }
if ($Crossfade -gt 0) { $cmd += @("--crossfade-ms", $Crossfade) }
if ($Extra -gt 0)     { $cmd += @("--extra-ms", $Extra) }

python @cmd
