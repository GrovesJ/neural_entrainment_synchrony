# auditory-entrainment

Project workspace for auditory entrainment modeling, stimuli generation, and analysis.

## Set up the notebook kernel on Windows

From the repository root, run PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_notebook.ps1
```

The script creates `.venv`, installs the model and notebook dependencies, and
registers the `Python (auditory-entrainment)` Jupyter kernel.

To activate the environment manually later:

```powershell
.\.venv\Scripts\Activate.ps1
```
