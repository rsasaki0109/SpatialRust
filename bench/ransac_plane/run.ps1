# CPU vs GPU RANSAC plane benchmark (Windows).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
python bench/ransac_plane/run.py @args
