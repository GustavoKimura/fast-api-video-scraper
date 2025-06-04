@echo off
cd /d %~dp0

echo Starting Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

echo Waiting for Docker to start...
:wait_loop
docker info >nul 2>&1
if errorlevel 1 (
  timeout /t 5 >nul
  goto wait_loop
)

set "SEARXNG_DIR=%~dp0searxng"

if not exist "%SEARXNG_DIR%\docker-compose.yml" (
  echo Error: Cannot find docker-compose.yml in "%SEARXNG_DIR%"
  pause
  exit /b 1
)

echo Starting SearxNG container...
pushd "%SEARXNG_DIR%"
docker compose up -d
popd

echo Server started on: http://192.168.0.103:8000
uvicorn main:app --host 192.168.0.103 --port 8000 --reload
