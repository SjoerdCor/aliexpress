# Wipe the local test environment: all process folders under instance/storage/
# and the SQLite database instance/app.db. The database is recreated empty by
# db.create_all() on the next app start; schools/admins must be re-created via
# `uv run flask schools create` / `uv run flask admins create`.
#
# Refuses to run while the server is up (open SQLite handle would block the
# delete and the app would keep stale state in memory). Use kill-server.ps1 first.
param(
    [switch]$Force  # skip the confirmation prompt
)

$root = $PSScriptRoot
$storage = Join-Path $root "instance\storage"
$database = Join-Path $root "instance\app.db"

# Guard: server must be down first.
$procs = Get-WmiObject Win32_Process |
    Where-Object { $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -match 'app\.py' }
if ($procs) {
    Write-Host "De server draait nog (PID $($procs.ProcessId -join ', ')). Stop die eerst met .\kill-server.ps1."
    exit 1
}

$targets = @()
if (Test-Path $storage) {
    $targets += Get-ChildItem $storage -Directory | Select-Object -ExpandProperty FullName
}
if (Test-Path $database) {
    $targets += $database
}

if (-not $targets) {
    Write-Host "Testomgeving is al leeg."
    exit 0
}

Write-Host "Dit verwijdert definitief:"
foreach ($t in $targets) {
    Write-Host "  $t"
}

if (-not $Force) {
    $answer = Read-Host "Doorgaan? (ja/nee)"
    if ($answer -ne "ja") {
        Write-Host "Geannuleerd."
        exit 0
    }
}

foreach ($t in $targets) {
    Remove-Item $t -Recurse -Force
}

Write-Host "Testomgeving leeggemaakt. Maak na de eerste start opnieuw scholen aan met:"
Write-Host "  uv run flask schools create <schoolcode> --naam <naam>"
