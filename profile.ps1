# PowerShell profile for this project
# Add this to your PowerShell profile for easy access

$PythonExe = "C:\Users\rasar\AppData\Local\Python\pythoncore-3.14-64\python.exe"

# Create aliases
Set-Alias -Name python -Value $PythonExe -Force
Set-Alias -Name py -Value $PythonExe -Force

# Function to run streamlit
function Start-Dashboard {
    Write-Host "Starting Streamlit dashboard..."
    & $PythonExe -m streamlit run src/dashboard.py
}

function Run-Tests {
    Write-Host "Running tests..."
    & $PythonExe -m pytest tests/test_metrics.py -v
}

function Evaluate-Baseline {
    Write-Host "Running baseline evaluation..."
    & $PythonExe -c "
from src.evaluator import Evaluator
from src.metrics import MetricsCalculator

evaluator = Evaluator()
print('Evaluator initialized')
print(f'Available test cases: {len(evaluator.list_test_cases())}')
"
}

Write-Host "Project aliases loaded: python, py, Start-Dashboard, Run-Tests, Evaluate-Baseline" -ForegroundColor Green
