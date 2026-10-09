@echo off
rem Bump VERSION, commit all changes and push to GitHub so other PCs can update.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\publish_update.ps1"
pause
