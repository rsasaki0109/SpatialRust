# Epic 60: public COPC validation harness (Windows).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
python bench/public_copc/run.py @args
