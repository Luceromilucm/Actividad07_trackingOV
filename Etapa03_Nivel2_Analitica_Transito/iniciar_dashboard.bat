@echo off
rem Abre el dashboard con el Python por defecto (python -m evita usar otra instalacion de streamlit)
cd /d "%~dp0"
python -m streamlit run dashboard.py
pause
