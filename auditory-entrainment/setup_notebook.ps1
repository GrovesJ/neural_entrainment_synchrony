$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$KernelName = "auditory-entrainment"
$KernelDisplayName = "Python (auditory-entrainment)"

if (-not (Test-Path $PythonPath)) {
    Write-Host "Creating virtual environment in $VenvPath..."
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $Launcher) {
        & py -3 -m venv $VenvPath
    }
    else {
        $Python = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $Python) {
            throw "Python 3 was not found. Install Python 3.10 or newer and try again."
        }
        & python -m venv $VenvPath
    }
}

if (-not (Test-Path $PythonPath)) {
    throw "The virtual environment was not created successfully: $PythonPath"
}

Write-Host "Upgrading pip..."
& $PythonPath -m pip install --upgrade pip

Write-Host "Installing project and notebook dependencies..."
& $PythonPath -m pip install --editable $ProjectRoot
& $PythonPath -m pip install `
    brian2 `
    ipykernel `
    ipywidgets `
    jupyter `
    matplotlib `
    numpy `
    pandas `
    scipy `
    tqdm

Write-Host "Registering Jupyter kernel..."
& $PythonPath -m ipykernel install --user `
    --name $KernelName `
    --display-name $KernelDisplayName

Write-Host ""
Write-Host "Setup complete."
Write-Host "Python: $PythonPath"
Write-Host "Kernel: $KernelDisplayName"
Write-Host ""
Write-Host "In VS Code, open notebooks\02_tuning.ipynb and select:"
Write-Host "  $KernelDisplayName"
