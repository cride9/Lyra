$ErrorActionPreference = 'Stop'
$lyraEnvironment = Join-Path $env:LOCALAPPDATA 'Lyra\venv'
if (-not (Test-Path -LiteralPath (Join-Path $lyraEnvironment 'Scripts\python.exe'))) {
    py -3.12 -m venv $lyraEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.12 first.' }
}
$lyraPython = Join-Path $lyraEnvironment 'Scripts\python.exe'
& $lyraPython -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $lyraPython -c "import subprocess,sys,tomllib; d=tomllib.load(open(sys.argv[1],'rb'))['project']['dependencies']; subprocess.check_call([sys.executable,'-m','pip','install',*d])" (Join-Path $PSScriptRoot 'pyproject.toml')
if ($LASTEXITCODE -ne 0) { throw 'Lyra dependency installation failed.' }
Write-Host 'Done. Launch Lyra with StartLyra.cmd'
