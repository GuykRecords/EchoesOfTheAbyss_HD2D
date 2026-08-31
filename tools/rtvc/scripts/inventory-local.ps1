<#
.SYNOPSIS
    D:\Claude\Project 配下の棚卸し。読み取り専用。何も消さない。

.DESCRIPTION
    リポジトリ移管後にローカルへ何が残っているかを一覧化し、
    「残す / 移す / 消す候補」に仕分けした表を出す。

    削除は一切行わない。消す候補は -ProposalOut で PowerShell スクリプトとして
    書き出せるので、中身を目で確認してから自分で実行すること。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\inventory-local.ps1
    powershell -ExecutionPolicy Bypass -File scripts\inventory-local.ps1 -ProposalOut cleanup-proposal.ps1
#>
[CmdletBinding()]
param(
    [string]$Root = 'D:\Claude\Project',
    [string]$RepoRtvc = (Join-Path (Split-Path -Parent $PSScriptRoot) ''),
    [string]$ProposalOut
)

$ErrorActionPreference = 'Stop'

function Get-DirSize {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { return 0 }
    return [math]::Round($sum / 1MB, 1)
}

Write-Host ''
Write-Host '=== ローカル棚卸し (読み取り専用) ===' -ForegroundColor Cyan
Write-Host "対象:        $Root"
Write-Host "リポジトリ側: $RepoRtvc"
Write-Host ''

if (-not (Test-Path -LiteralPath $Root)) {
    Write-Warning "$Root が見つかりません。-Root で正しいパスを指定してください。"
    exit 1
}

# --- 既知のディレクトリの扱いを宣言しておく（判断の根拠を残すため） -------------
$known = @(
    @{ Name = '.venv';     Verdict = '残す';     Why = '計測用 venv (numpy 2.x / torch cu128)。RVC は絶対に入れない' }
    @{ Name = '.venv-rvc'; Verdict = '残す';     Why = 'RVC 専用 venv (numpy 1.23.5 + fairseq)。無ければこれから作る' }
    @{ Name = 'RVC';       Verdict = '残す';     Why = 'RVC 本体の clone。リポジトリには入れない（巨大 & 別ライセンス）' }
    @{ Name = 'rtvc';      Verdict = '要判断';   Why = 'リポジトリ tools/rtvc へ移管済み。差分が無ければ退避してよい' }
)

$rows = @()
foreach ($item in Get-ChildItem -LiteralPath $Root -Force -ErrorAction SilentlyContinue) {
    $match = $known | Where-Object { $_.Name -eq $item.Name } | Select-Object -First 1
    $rows += [pscustomobject]@{
        Name    = $item.Name
        Type    = if ($item.PSIsContainer) { 'dir' } else { 'file' }
        SizeMB  = if ($item.PSIsContainer) { Get-DirSize $item.FullName } else { [math]::Round($item.Length / 1MB, 2) }
        Updated = $item.LastWriteTime.ToString('yyyy-MM-dd')
        Verdict = if ($match) { $match.Verdict } else { '未分類' }
        Why     = if ($match) { $match.Why } else { '心当たりが無ければ中身を確認してから判断' }
    }
}

$rows | Sort-Object -Property @{Expression = 'SizeMB'; Descending = $true } |
    Format-Table Name, Type, SizeMB, Updated, Verdict, Why -AutoSize -Wrap

# --- 旧 rtvc とリポジトリ版の差分 ---------------------------------------------
$oldRtvc = Join-Path $Root 'rtvc'
Write-Host ''
Write-Host '=== 旧 rtvc とリポジトリ版の比較 ===' -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $oldRtvc)) {
    Write-Host "$oldRtvc は存在しません。移管済みか、まだ作っていないかのどちらかです。"
} else {
    $localOnly = @()
    foreach ($f in Get-ChildItem -LiteralPath $oldRtvc -Recurse -File -Force -ErrorAction SilentlyContinue) {
        $rel = $f.FullName.Substring($oldRtvc.Length).TrimStart('\')
        if ($rel -match '^(__pycache__|\.pytest_cache)') { continue }

        # リポジトリ側は rtvc/ パッケージ配下に置き直してある
        $candidates = @(
            (Join-Path $RepoRtvc $rel),
            (Join-Path (Join-Path $RepoRtvc 'rtvc') $rel)
        )
        $repoFile = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

        if (-not $repoFile) {
            $localOnly += $rel
            Write-Host ("  [ローカルのみ] {0}" -f $rel) -ForegroundColor Yellow
        } else {
            $a = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash
            $b = (Get-FileHash -LiteralPath $repoFile   -Algorithm SHA256).Hash
            if ($a -eq $b) {
                Write-Host ("  [一致]         {0}" -f $rel) -ForegroundColor DarkGray
            } else {
                Write-Host ("  [差分あり]     {0}" -f $rel) -ForegroundColor Magenta
                Write-Host ("                 -> {0}" -f $repoFile) -ForegroundColor DarkGray
            }
        }
    }
    Write-Host ''
    if ($localOnly.Count -gt 0) {
        Write-Warning "ローカルにしか無いファイルが $($localOnly.Count) 件あります。消す前に中身を確認してください。"
    } else {
        Write-Host 'ローカル固有のファイルはありません。' -ForegroundColor Green
    }
}

# --- 削除案の書き出し（実行はしない） -----------------------------------------
if ($ProposalOut) {
    $stamp = Get-Date -Format 'yyyyMMdd'
    $lines = @(
        '# 自動生成された「削除案」。実行前に必ず中身を読むこと。',
        '# このスクリプトは削除ではなく退避 (rename) を行う。',
        '# 1〜2 週間動かして問題が無ければ、退避先を手で消す。',
        '',
        '$ErrorActionPreference = ''Stop''',
        "`$old = '$oldRtvc'",
        "`$archive = '$oldRtvc._archived_$stamp'",
        'if (Test-Path -LiteralPath $old) {',
        '    Rename-Item -LiteralPath $old -NewName (Split-Path -Leaf $archive)',
        '    Write-Host "退避しました: $archive"',
        '} else {',
        '    Write-Host "$old は存在しません。何もしません。"',
        '}'
    )
    Set-Content -LiteralPath $ProposalOut -Value $lines -Encoding UTF8
    Write-Host ''
    Write-Host "削除案を書き出しました: $ProposalOut" -ForegroundColor Cyan
    Write-Host '中身を読んでから、自分で実行してください。このスクリプトは実行しません。'
}

Write-Host ''
Write-Host '棚卸し完了。このスクリプトは何も変更していません。' -ForegroundColor Green
