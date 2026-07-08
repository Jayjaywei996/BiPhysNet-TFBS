$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $PROJECT_ROOT
$env:PYTHONPATH = "$PROJECT_ROOT\src;$env:PYTHONPATH"

Write-Host "[1/5] Checking hard-coded local paths outside legacy/..."
$matches = Get-ChildItem "$PROJECT_ROOT" -Recurse -File |
  Where-Object { $_.FullName -notmatch "\\legacy\\" -and $_.FullName -notmatch "\\.git\\" -and $_.Name -notin @("check_project.ps1", "check_project.sh") } |
  Select-String -Pattern "C:\\Users\\13971\\Music", "/home/wjw" -ErrorAction SilentlyContinue
if ($matches) {
  Write-Host "Found hard-coded paths:" -ForegroundColor Yellow
  $matches | ForEach-Object { Write-Host $_ }
} else {
  Write-Host "No old local hard-coded paths found outside legacy/."
}

Write-Host "[2/5] Checking for private PDFs in GitHub-ready tree..."
$pdfFiles = Get-ChildItem "$PROJECT_ROOT" -Recurse -File -Include *.pdf | Where-Object { $_.FullName -notmatch "\\.git\\" }
if ($pdfFiles) {
  Write-Host "PDF files were found. Keep manuscript/patent drafts private before publishing." -ForegroundColor Yellow
  $pdfFiles | ForEach-Object { Write-Host $_.FullName }
} else {
  Write-Host "No PDF files found."
}

Write-Host "[3/5] Compiling Python files..."
$pyFiles = @(Get-ChildItem "$PROJECT_ROOT\src\biphysnet\*.py" | ForEach-Object { $_.FullName }) + @(Get-ChildItem "$PROJECT_ROOT\tools\*.py" | ForEach-Object { $_.FullName })
python -m py_compile $pyFiles

Write-Host "[4/5] Running tests..."
python -m pytest tests

Write-Host "[5/5] Git status..."
if (-not (Test-Path ".git")) { git init | Out-Null }
git status --short
Pop-Location
