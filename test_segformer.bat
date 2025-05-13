@echo off
SETLOCAL ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

:: Get arguments
SET "CKPT=%~1"
SET "INPUT=%~2"
SET "OUTPUT=%~3"

:: Clear previous log
echo. > build_logs.txt

(
    echo [*] Checkpoint   = %CKPT%
    echo [*] Input video  = %INPUT%
    echo [*] Output video = %OUTPUT%

    :: Build Docker image if it doesn't exist
    docker image inspect segformer >nul 2>&1
    IF %ERRORLEVEL% NEQ 0 (
        echo [*] Building Docker image 'segformer'...
        docker build -t segformer .
    ) ELSE (
        echo [✓] Docker image 'segformer' already exists. Skipping build.
    )

    :: Run Docker container
     docker run --rm -it --gpus all ^
      -v "%cd%":/home/segformer_docker/seg ^
      segformer python3 /home/segformer_docker/seg/segformer_script.py ^
      "/home/segformer_docker/seg/%CKPT%" ^
      "/home/segformer_docker/seg/%INPUT%" ^
      "/home/segformer_docker/seg/%OUTPUT%"
) >> build_logs.txt 2>&1