@echo off
setlocal
cd /d "%~dp0.."
set BASE_DIR=%CD%
set SPARK_HOME=%BASE_DIR%\.venv\Lib\site-packages\pyspark
set JAVA_HOME=C:\Program Files\Java\jdk-17
set PATH=C:\Program Files\Java\jdk-17\bin;%PATH%
set HADOOP_HOME=%BASE_DIR%\.runtime\hadoop
set PATH=%HADOOP_HOME%\bin;%PATH%

echo ==============================================================================
echo [1/3] Starting Spark Master on 127.0.0.1:7077 (Web UI: 8080)...
echo ==============================================================================
start "Spark-Master" /min "%SPARK_HOME%\bin\spark-class.cmd" org.apache.spark.deploy.master.Master --host 127.0.0.1 --port 7077 --webui-port 8080

ping -n 5 127.0.0.1 >nul

echo [2/3] Starting Spark Worker 1 (Web UI: 8081)...
start "Spark-Worker-1" /min "%SPARK_HOME%\bin\spark-class.cmd" org.apache.spark.deploy.worker.Worker spark://127.0.0.1:7077 --webui-port 8081 -c 2 -m 2G

echo [3/3] Starting Spark Worker 2 (Web UI: 8082)...
start "Spark-Worker-2" /min "%SPARK_HOME%\bin\spark-class.cmd" org.apache.spark.deploy.worker.Worker spark://127.0.0.1:7077 --webui-port 8082 -c 2 -m 2G

ping -n 4 127.0.0.1 >nul

echo.
echo ==============================================================================
echo [OK] Spark Standalone Cluster Started!
echo Master Web UI   : http://localhost:8080
echo Worker 1 Web UI : http://localhost:8081
echo Worker 2 Web UI : http://localhost:8082
echo ==============================================================================
