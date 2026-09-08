# Launch realtime voice conversion with the settings we settled on by ear.
#
#   .\go-live.ps1                  monitor in your own ears, until Ctrl+C
#   .\go-live.ps1 -Discord         send to VB-CABLE, for Discord and OBS
#   .\go-live.ps1 -Duration 30     stop after 30 seconds
#   .\go-live.ps1 -Record take.wav also save the raw input, to replay offline
#
# ASCII only, on purpose. Windows PowerShell 5.1 reads a .ps1 as the system
# ANSI code page (cp932 here), so BOM-less UTF-8 Japanese arrives as mojibake
# and takes the parser down with it. An earlier script in this project died
# exactly that way. Comments in this file stay in English.
#
# Every default below was measured, not guessed -- see docs/HANDOVER.md.

[CmdletBinding()]
param(
    # Route the converted voice to VB-CABLE instead of your own ears.
    [switch]$Discord,

    [string]$InDevice  = "INZONE Buds - Chat",
    [string]$OutDevice = "INZONE Buds - Game",

    # 0 means run until Ctrl+C.
    [double]$Duration = 0,

    # Pitch +12 makes it another person at all -- the single biggest knob.
    # Formant went to 1.5 on a deliberately-spoken take and back to 0 on
    # ordinary conversation, where it added warble. Ordinary conversation is
    # what this is for.
    [int]$Key        = 12,
    [double]$Formant = 0,

    # Index off sounded the same on a careful take, so it was switched off.
    # That was measured on the wrong material; with the faiss nprobe fixed it
    # is cheap (about 2ms) and on by default now.
    [double]$IndexRate = 0.75,

    # Past context. Costs compute, not latency, and more of it measurably
    # helped -- the best of the three window changes tried against RVC's own
    # file conversion. 0 leaves the engine default (2500).
    [double]$Extra = 5000,

    [string]$ModelGlob = "D:\Claude\Project\RVC\assets\weights\ena_e150_s*.pth",
    [string]$IndexGlob = "D:\Claude\Project\RVC\assets\indices\*ena*.index",
    [string]$Venv      = "D:\Claude\Project\.venv-rvc",

    # Save the captured input, so the same take can be replayed through other
    # settings offline. That is how every setting here was decided.
    [string]$Record = ""
)

$ErrorActionPreference = "Stop"

if ($Discord) { $OutDevice = "CABLE Input" }

# Activate the venv only if this window does not already have one. Opening a
# fresh PowerShell and forgetting this is the single most common stumble.
if (-not $env:VIRTUAL_ENV) {
    $activate = Join-Path $Venv "Scripts\Activate.ps1"
    if (-not (Test-Path $activate)) {
        throw "venv not found: $activate  (pass -Venv if it moved)"
    }
    & $activate
}

# Run from the repository, wherever the script was invoked from.
Set-Location $PSScriptRoot

$model = (Get-ChildItem $ModelGlob -ErrorAction SilentlyContinue |
          Sort-Object Name | Select-Object -Last 1).FullName
if (-not $model) {
    throw "no speaker model matched: $ModelGlob  (pass -ModelGlob to point elsewhere)"
}

$cmd = @(
    "-m", "rtvc.realtime", "--engine", "rvc",
    "--rvc-model", $model,
    "--rvc-key", $Key,
    "--rvc-formant", $Formant,
    "--host-api", "WASAPI",
    "--in-device", $InDevice,
    "--out-device", $OutDevice
)
if ($IndexRate -gt 0) {
    $index = (Get-ChildItem $IndexGlob -ErrorAction SilentlyContinue |
              Sort-Object Name | Select-Object -Last 1).FullName
    if (-not $index) { throw "index rate is $IndexRate but no index matched: $IndexGlob" }
    $cmd += @("--rvc-index", $index, "--rvc-index-rate", $IndexRate)
}

if ($Extra -gt 0)    { $cmd += @("--extra-ms", $Extra) }
if ($Duration -gt 0) { $cmd += @("--duration", $Duration) }
if ($Record)         { $cmd += @("--record-in", $Record) }

Write-Host "model : $(Split-Path $model -Leaf)"
Write-Host "in    : $InDevice"
Write-Host "out   : $OutDevice"
Write-Host "note  : watch under/over/drop. They must stay at 0."
Write-Host ""

python @cmd
