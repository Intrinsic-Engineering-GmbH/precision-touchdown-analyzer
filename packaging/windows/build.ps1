<#
.SYNOPSIS
  Build the Windows installer: dist\PTA-Setup-<version>.exe

.DESCRIPTION
  Run from the repository root, inside the project's virtual environment
  (the one with the [ui,analysis] extras and pyinstaller installed):

      pip install -e ".[ui,analysis]" pyinstaller
      .\packaging\windows\build.ps1

  Steps: ffmpeg -> icon -> PyInstaller application folder -> zip ->
  PyInstaller setup program with the zip inside. ffmpeg.exe and ffprobe.exe
  are downloaded once into vendor\ffmpeg\ (fetch_ffmpeg.py: gyan.dev
  "essentials" release, checksum verified) and installed as tools\ next to
  the program, so nothing needs to be on PATH at the airfield. -NoFfmpeg
  skips that; the installed program then looks for ffmpeg on PATH and the
  control window says so if it is missing.

.PARAMETER NoFfmpeg
  Do not download or bundle ffmpeg.
#>
param([switch]$NoFfmpeg)
# "Continue": PyInstaller logs to stderr, which Windows PowerShell 5.1 would
# otherwise turn into a terminating error; exit codes are checked instead.
$ErrorActionPreference = "Continue"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

if (-not $NoFfmpeg) {
  Write-Host "== ffmpeg"
  & $python packaging\windows\fetch_ffmpeg.py
  if ($LASTEXITCODE -ne 0) { throw "fetching ffmpeg failed (use -NoFfmpeg to build without it)" }
}

Write-Host "== icon"
& $python packaging\make_icon.py packaging\out

Write-Host "== application folder"
if (Test-Path build\dist) { Remove-Item build\dist -Recurse -Force }
& $python -m PyInstaller packaging\windows\app.spec --noconfirm --distpath build\dist --workpath build\work --log-level WARN
if ($LASTEXITCODE -ne 0) { throw "PyInstaller (app) failed" }

$app = "build\dist\PTA"
Copy-Item packaging\out\icon.ico "$app\icon.ico" -Force
Copy-Item packaging\out\icon.png "$app\icon.png" -Force
Copy-Item LICENSE "$app\LICENSE.txt" -Force

Write-Host "== smoke test of the folder"
& "$app\touchdown-analyzer.exe" --version
if ($LASTEXITCODE -ne 0) { throw "the built CLI does not run" }
if (-not $NoFfmpeg) {
  # PyInstaller 6 puts data files under _internal\; paths.bundled_tool looks there too.
  $bundled = Get-ChildItem $app -Recurse -Filter ffmpeg.exe | Select-Object -First 1
  if (-not $bundled) { throw "ffmpeg.exe is missing from the application folder" }
  & $bundled.FullName -version | Select-Object -First 1
  if ($LASTEXITCODE -ne 0) { throw "the bundled ffmpeg does not run" }
}

Write-Host "== app.zip"
if (Test-Path build\app.zip) { Remove-Item build\app.zip -Force }
# Python's zipfile rather than Compress-Archive, which now and then fails
# silently on a folder the virus scanner is still busy with.
& $python -c "import shutil; shutil.make_archive('build/app', 'zip', r'$app')"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path build\app.zip)) { throw "app.zip was not written" }

Write-Host "== setup program"
New-Item -ItemType Directory -Force dist | Out-Null
& $python -m PyInstaller packaging\windows\setup.spec --noconfirm --distpath dist --workpath build\work-setup --log-level WARN
if ($LASTEXITCODE -ne 0) { throw "PyInstaller (setup) failed" }

Get-ChildItem dist\PTA-Setup-*.exe | ForEach-Object {
  Write-Host ("== done: {0}  ({1:N1} MB)" -f $_.FullName, ($_.Length / 1MB))
}
