@echo off
rem Demo del Nivel 1: YOLO + ByteTrack sobre el video por defecto. Q o ESC termina y genera el analisis.
cd /d "%~dp0.."
python demo_nivel1\demo_yolo.py %*
pause
