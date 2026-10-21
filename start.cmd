@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\windows\start.ps1" -SourceRoot "%~dp0." %*
exit /b %ERRORLEVEL%
