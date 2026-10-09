@echo off
rem Build CellCounter.exe with the C# compiler that ships with Windows (.NET Framework 4) - nothing to install.
cd /d "%~dp0"
"%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe" /nologo /target:exe /optimize+ /codepage:65001 ^
  /out:CellCounter.exe /reference:System.IO.Compression.dll /reference:System.IO.Compression.FileSystem.dll ^
  CellCounterLauncher.cs
if errorlevel 1 (echo Build failed & exit /b 1)
echo Built: %~dp0CellCounter.exe
