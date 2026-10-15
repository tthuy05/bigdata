@echo off
setlocal
echo ==============================================================================
echo Stopping all Spark Master and Worker processes...
echo ==============================================================================

taskkill /F /FI "WINDOWTITLE eq Spark-Master*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Spark-Worker-1*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Spark-Worker-2*" >nul 2>&1

powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*org.apache.spark.deploy.master.Master*' -or $_.CommandLine -like '*org.apache.spark.deploy.worker.Worker*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

echo [OK] All Spark cluster processes stopped.
echo ==============================================================================
