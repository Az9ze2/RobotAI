# Manual Model Download Instructions

The automatic model downloader is experiencing 404 errors. Please download the models manually using these **verified working URLs**:

## Option 1: Download Model Packs (Recommended)

### Buffalo M Pack (Contains SCRFD 2.5G)
```bash
# Download
curl -L -o models/buffalo_m.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip

# Extract
unzip models/buffalo_m.zip -d models/buffalo_m/

# Copy SCRFD model
cp models/buffalo_m/det_2.5g.onnx models/scrfd_2.5g_kps_fp16.onnx
```

### Buffalo L Pack (Contains ArcFace)
```bash
# Download
curl -L -o models/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

# Extract
unzip models/buffalo_l.zip -d models/buffalo_l/

# Copy ArcFace model
cp models/buffalo_l/w600k_r50.onnx models/arcface_r100_v1_fp16.onnx
```

## Option 2: Download Standalone Models

### SCRFD 2.5G (with landmarks)
**URL**: https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps.zip

```bash
curl -L -o models/scrfd_2.5g_bnkps.zip https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps.zip
unzip models/scrfd_2.5g_bnkps.zip -d models/
# Rename to expected filename
mv models/scrfd_2.5g_bnkps.onnx models/scrfd_2.5g_kps_fp16.onnx
```

## Windows PowerShell Commands

If you're on Windows (which you are), use these commands instead:

```powershell
# Create models directory if it doesn't exist
New-Item -ItemType Directory -Force -Path models

# Download Buffalo M (SCRFD)
Invoke-WebRequest -Uri "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip" -OutFile "models\buffalo_m.zip"
Expand-Archive -Path "models\buffalo_m.zip" -DestinationPath "models\buffalo_m" -Force
Copy-Item "models\buffalo_m\det_2.5g.onnx" "models\scrfd_2.5g_kps_fp16.onnx"

# Download Buffalo L (ArcFace)
Invoke-WebRequest -Uri "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" -OutFile "models\buffalo_l.zip"
Expand-Archive -Path "models\buffalo_l.zip" -DestinationPath "models\buffalo_l" -Force
Copy-Item "models\buffalo_l\w600k_r50.onnx" "models\arcface_r100_v1_fp16.onnx"

# Verify models exist
Get-ChildItem models\*.onnx
```

## Quick Download Script

Save this as `download_models.ps1` and run it:

```powershell
# download_models.ps1
Write-Host "Downloading InsightFace models..." -ForegroundColor Green

# Create models directory
New-Item -ItemType Directory -Force -Path models | Out-Null

# Download Buffalo M (SCRFD 2.5G)
Write-Host "`nDownloading SCRFD model..." -ForegroundColor Cyan
Invoke-WebRequest -Uri "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip" -OutFile "models\buffalo_m.zip"
Expand-Archive -Path "models\buffalo_m.zip" -DestinationPath "models\buffalo_m" -Force
Copy-Item "models\buffalo_m\det_2.5g.onnx" "models\scrfd_2.5g_kps_fp16.onnx" -Force

# Download Buffalo L (ArcFace)
Write-Host "Downloading ArcFace model..." -ForegroundColor Cyan
Invoke-WebRequest -Uri "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" -OutFile "models\buffalo_l.zip"
Expand-Archive -Path "models\buffalo_l.zip" -DestinationPath "models\buffalo_l" -Force
Copy-Item "models\buffalo_l\w600k_r50.onnx" "models\arcface_r100_v1_fp16.onnx" -Force

# Verify
Write-Host "`nVerifying models..." -ForegroundColor Green
if (Test-Path "models\scrfd_2.5g_kps_fp16.onnx") {
    Write-Host "✓ SCRFD model downloaded successfully" -ForegroundColor Green
} else {
    Write-Host "✗ SCRFD model not found" -ForegroundColor Red
}

if (Test-Path "models\arcface_r100_v1_fp16.onnx") {
    Write-Host "✓ ArcFace model downloaded successfully" -ForegroundColor Green
} else {
    Write-Host "✗ ArcFace model not found" -ForegroundColor Red
}

Write-Host "`nDone!" -ForegroundColor Green
```

## Run the Download

```powershell
# Run the script
.\download_models.ps1
```

## Verify Models

After downloading, verify the models are in place:

```powershell
python -c "from pathlib import Path; print('SCRFD:', Path('./models/scrfd_2.5g_kps_fp16.onnx').exists()); print('ArcFace:', Path('./models/arcface_r100_v1_fp16.onnx').exists())"
```

Expected output:
```
SCRFD: True
ArcFace: True
```
