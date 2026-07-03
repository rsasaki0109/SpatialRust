# Euclidean cluster CPU vs GPU benchmark (Windows).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
python bench/euclidean_cluster/run.py @args
