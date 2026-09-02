# Stop only this repository's app.py instances (Flask reloader spawns parent +
# child processes; killing only the port owner leaves the parent to respawn a
# new child).
$appPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'app.py'))
$appPathPattern = [regex]::Escape($appPath)
$procs = Get-WmiObject Win32_Process |
    Where-Object {
        $_.Name -match '^pythonw?\.exe$' -and
        $_.CommandLine -and
        $_.CommandLine -match "(?i)(?:(?<=\x22)$appPathPattern(?=\x22)|(?<![\w.-])$appPathPattern(?![\w.-]))"
    }

if (-not $procs) {
    Write-Host "Geen app.py-processen gevonden."
    exit 0
}

$ids = $procs | Select-Object -ExpandProperty ProcessId
foreach ($id in $ids) {
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Milliseconds 500

$remaining = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "Waarschuwing: poort 5000 nog bezet door PID $($remaining.OwningProcess)."
} else {
    Write-Host "Server gestopt."
}
