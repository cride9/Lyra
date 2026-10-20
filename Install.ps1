$ErrorActionPreference = 'Stop'
$lyraEnvironment = Join-Path $env:LOCALAPPDATA 'Lyra\venv'
if (-not (Test-Path -LiteralPath (Join-Path $lyraEnvironment 'Scripts\python.exe'))) {
    py -3.12 -m venv $lyraEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'A 64 bites Python 3.12 telepítése szükséges.' }
}
$lyraPython = Join-Path $lyraEnvironment 'Scripts\python.exe'
& $lyraPython -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'A CUDA PyTorch telepítése nem sikerült.' }
& $lyraPython -c "import subprocess,sys,tomllib; d=tomllib.load(open(sys.argv[1],'rb'))['project']['dependencies']; subprocess.check_call([sys.executable,'-m','pip','install',*d])" (Join-Path $PSScriptRoot 'pyproject.toml')
if ($LASTEXITCODE -ne 0) { throw 'A Lyra függőségeinek telepítése nem sikerült.' }
Write-Host 'Kész. Indítás: StartLyra.cmd'
