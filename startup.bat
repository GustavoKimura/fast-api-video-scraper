@echo off
cd /d %~dp0

echo Servidor iniciado em: http://192.168.0.103:8000

uvicorn main:app --host 192.168.0.103 --port 8000 --reload
