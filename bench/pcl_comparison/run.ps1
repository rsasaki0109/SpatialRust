[CmdletBinding()]
param(
    [int]$Points = 200000,
    [string]$VcpkgRoot = "C:\vcpkg",
    [string]$Msys2Root = "C:\msys64",
    [string]$Triplet = "x64-mingw-dynamic-release-bigobj"
)

$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $Here "..\..")).Path
$BuildDir = Join-Path $Root "target\pcl-bench"
$Pcd = Join-Path $BuildDir "bench_cloud.pcd"
$SrOut = Join-Path $BuildDir "sr_out.csv"
$PclOut = Join-Path $BuildDir "pcl_out.csv"
$OverlayTriplets = Join-Path $Here "vcpkg-triplets"

$MingwBin = Join-Path $Msys2Root "ucrt64\bin"
$MsysUsrBin = Join-Path $Msys2Root "usr\bin"
$VcpkgBin = Join-Path $VcpkgRoot "installed\$Triplet\bin"
$CMake = Join-Path $MingwBin "cmake.exe"
$Cc = Join-Path $MingwBin "gcc.exe"
$Cxx = Join-Path $MingwBin "g++.exe"

$PythonCandidates = @(
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path $Root ".venv\bin\python"),
    "python"
)
$Python = $PythonCandidates | Where-Object { Get-Command $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
if (-not $Python) {
    throw "python not found"
}

$CargoCandidates = @(
    (Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"),
    "cargo"
)
$Cargo = $CargoCandidates | Where-Object { Get-Command $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
if (-not $Cargo) {
    throw "cargo not found"
}

New-Item -ItemType Directory -Force $BuildDir | Out-Null
$env:PATH = "$VcpkgBin;$MingwBin;$MsysUsrBin;$env:PATH"

Write-Host "== generating $Points-point cloud =="
& $Python (Join-Path $Here "gen_cloud.py") --points $Points --out $Pcd

Write-Host "== building SpatialRust bench =="
& $Cargo build --release --manifest-path (Join-Path $Root "Cargo.toml") -p spatialrust --example bench_ops --features mvp,filter-outlier

Write-Host "== configuring PCL bench =="
& $CMake -S $Here -B $BuildDir -G Ninja `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DCMAKE_TOOLCHAIN_FILE=$(Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake")" `
    "-DVCPKG_TARGET_TRIPLET=$Triplet" `
    "-DVCPKG_OVERLAY_TRIPLETS=$OverlayTriplets" `
    "-DCMAKE_C_COMPILER=$Cc" `
    "-DCMAKE_CXX_COMPILER=$Cxx"

Write-Host "== building PCL bench =="
& $CMake --build $BuildDir --config Release

$SrExe = Join-Path $Root "target\release\examples\bench_ops.exe"
if (-not (Test-Path $SrExe)) {
    $SrExe = Join-Path $Root "target\release\examples\bench_ops"
}
$PclExe = Join-Path $BuildDir "pcl_bench.exe"
if (-not (Test-Path $PclExe)) {
    $PclExe = Join-Path $BuildDir "pcl_bench"
}

Write-Host "== running SpatialRust =="
$SrLines = & $SrExe $Pcd
$SrLines | Set-Content -Encoding utf8 $SrOut

Write-Host "== running PCL =="
$PclLines = & $PclExe $Pcd
$PclLines | Set-Content -Encoding utf8 $PclOut

$SrRows = Import-Csv -Path $SrOut -Header Operation, Seconds, OutputPoints
$PclRows = Import-Csv -Path $PclOut -Header Operation, Seconds, OutputPoints
$PclByOperation = @{}
foreach ($Row in $PclRows) {
    $PclByOperation[$Row.Operation] = $Row
}

Write-Host ""
"{0,-30} {1,14} {2,14} {3,10}" -f "operation", "SpatialRust(s)", "PCL(s)", "speedup"
"{0,-30} {1,14} {2,14} {3,10}" -f "------------------------------", "--------------", "--------------", "----------"
foreach ($Row in $SrRows) {
    $PclRow = $PclByOperation[$Row.Operation]
    if ($PclRow) {
        $SrSeconds = [double]$Row.Seconds
        $PclSeconds = [double]$PclRow.Seconds
        $Speedup = if ($SrSeconds -gt 0.0) { "{0:F2}x" -f ($PclSeconds / $SrSeconds) } else { "n/a" }
        "{0,-30} {1,14} {2,14} {3,10}" -f $Row.Operation, $Row.Seconds, $PclRow.Seconds, $Speedup
    } else {
        "{0,-30} {1,14} {2,14} {3,10}" -f $Row.Operation, $Row.Seconds, "n/a", "n/a"
    }
}
