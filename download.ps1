Set-Location $PSScriptRoot
$ErrorActionPreference = "Stop"

$root = Join-Path $PSScriptRoot "backend\data\models"
$downloads = Join-Path $PSScriptRoot "downloads"

function Save-File([string]$Dest, [string]$Url) {
    $folder = Split-Path $Dest
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    if ((Test-Path $Dest) -and (Get-Item $Dest).Length -gt 0) {
        Write-Host "있음  $Dest"
        return
    }

    $part = "$Dest.part"
    Write-Host "받는 중  $Dest"
    & curl.exe -L --fail --retry 3 -o $part $Url
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path $part) { Remove-Item $part -Force }
        throw "다운로드 실패: $Dest"
    }
    Move-Item -Force $part $Dest
}

$files = @(
    @{
        Dir = "sam3_point"
        Name = "vision_encoder_q4f16.onnx"
        Url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/vision_encoder_q4f16.onnx"
    },
    @{
        Dir = "sam3_point"
        Name = "vision_encoder_q4f16.onnx_data"
        Url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/vision_encoder_q4f16.onnx_data"
    },
    @{
        Dir = "sam3_point"
        Name = "prompt_encoder_mask_decoder_int8.onnx"
        Url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder_int8.onnx"
    },
    @{
        Dir = "sam3_point"
        Name = "prompt_encoder_mask_decoder_int8.onnx_data"
        Url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder_int8.onnx_data"
    },
    @{
        Dir = "sam3_text"
        Name = "vision_encoder_int4.onnx"
        Url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/vision_encoder_int4.onnx"
    },
    @{
        Dir = "sam3_text"
        Name = "vision_encoder_int4.onnx.data"
        Url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/vision_encoder_int4.onnx.data"
    },
    @{
        Dir = "sam3_text"
        Name = "text_encoder_int4.onnx"
        Url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/text_encoder_int4.onnx"
    },
    @{
        Dir = "sam3_text"
        Name = "text_encoder_int4.onnx.data"
        Url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/text_encoder_int4.onnx.data"
    },
    @{
        Dir = "sam3_text"
        Name = "decoder_int4.onnx"
        Url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/decoder_int4.onnx"
    },
    @{
        Dir = "sam3_text"
        Name = "vocab.json"
        Url = "https://github.com/jamjamjon/assets/releases/download/sam3/vocab.json"
    },
    @{
        Dir = "sam3_text"
        Name = "merges.txt"
        Url = "https://github.com/jamjamjon/assets/releases/download/sam3/merges.txt"
    }
)

foreach ($file in $files) {
    Save-File (Join-Path $root (Join-Path $file.Dir $file.Name)) $file.Url
}

$cudnnZip = Join-Path $downloads "cudnn\cudnn-windows-x86_64-9.24.0.43_cuda12-archive.zip"
Save-File $cudnnZip "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.24.0.43_cuda12-archive.zip"

$cudnnRoot = Join-Path $downloads "cudnn\v9.24"
$cudnnLib = Join-Path $cudnnRoot "lib\12.8\x64\cudnn.lib"
if (-not (Test-Path $cudnnLib)) {
    $extract = Join-Path $downloads "cudnn\extract"
    New-Item -ItemType Directory -Force -Path $extract | Out-Null
    Write-Host "푸는 중  $cudnnZip"
    & tar.exe -xf $cudnnZip -C $extract
    if ($LASTEXITCODE -ne 0) { throw "cuDNN 압축을 풀지 못했습니다." }
    $src = Join-Path $extract "cudnn-windows-x86_64-9.24.0.43_cuda12-archive"
    New-Item -ItemType Directory -Force -Path (Join-Path $cudnnRoot "include\12.8") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $cudnnRoot "lib\12.8\x64") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $cudnnRoot "bin\12.8\x64") | Out-Null
    Copy-Item (Join-Path $src "include\*") (Join-Path $cudnnRoot "include\12.8") -Force
    Copy-Item (Join-Path $src "lib\x64\*") (Join-Path $cudnnRoot "lib\12.8\x64") -Force
    Copy-Item (Join-Path $src "bin\x64\*") (Join-Path $cudnnRoot "bin\12.8\x64") -Force
}

$cudnnInstall = "C:\Program Files\NVIDIA\CUDNN\v9.24"
$cudnnInstallLib = Join-Path $cudnnInstall "lib\12.8\x64\cudnn.lib"
if (-not (Test-Path $cudnnInstallLib)) {
    Write-Host "설치 중  $cudnnInstall"
    $installScript = Join-Path $env:TEMP "install-cudnn.ps1"
    @(
        '$ErrorActionPreference = "Stop"'
        "`$from = '$cudnnRoot'"
        "`$toRoot = '$cudnnInstall'"
        'foreach ($dir in @("include\12.8", "lib\12.8\x64", "bin\12.8\x64")) {'
        '    $to = Join-Path $toRoot $dir'
        '    New-Item -ItemType Directory -Force -Path $to | Out-Null'
        '    Copy-Item (Join-Path (Join-Path $from $dir) "*") $to -Force'
        '}'
    ) | Set-Content -Path $installScript -Encoding UTF8
    $proc = Start-Process powershell -Verb RunAs -Wait -PassThru -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $installScript
    if ($proc.ExitCode -ne 0 -or -not (Test-Path $cudnnInstallLib)) {
        throw "cuDNN을 $cudnnInstall 에 복사하지 못했습니다."
    }
}

Save-File (Join-Path $downloads "db\SQL2022-SSEI-Expr.exe") "https://download.microsoft.com/download/29654887-7cde-4397-bba3-d7f087970845/SQL2022-SSEI-Expr.exe"

Write-Host "다운로드가 끝났습니다."
Write-Host "cuDNN: downloads\cudnn\v9.24"
Write-Host "SQL Server 2022 Express 설치 파일: downloads\db\SQL2022-SSEI-Expr.exe"
