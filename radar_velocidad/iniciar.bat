@echo off
cd /d "%~dp0"
echo Instalando dependencias (solo la primera vez tarda)...
python -m pip install -q -r requirements.txt
python app.py
if errorlevel 1 pause
