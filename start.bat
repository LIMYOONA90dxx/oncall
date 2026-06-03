@echo off
chcp 936 >nul
setlocal enabledelayedexpansion

echo ====================================
echo 启动 SuperBizAgent 服务
echo ====================================
echo.

REM ==================== 错误捕获 ====================
set "ERROR_LOG=startup_error.log"
del "%ERROR_LOG%" 2>nul

echo [1/6] 检查包管理器...
where uv >nul 2>&1
if errorlevel 1 (
    echo [信息] uv 未安装，将使用传统 pip 方式
    set USE_UV=0
) else (
    echo [成功] 检测到 uv 包管理器
    set USE_UV=1
)
echo.

echo [2/6] 配置 Python 版本...
if not exist .python-version (
    echo 3.13> .python-version
)
set /p PYTHON_VERSION=<.python-version
echo [信息] 当前Python版本: !PYTHON_VERSION!
echo.

echo [3/6] 检查虚拟环境...
if not exist .venv\Scripts\python.exe (
    echo [信息] 正在创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ==============================
        echo 【错误】Python 不可用！！！
        echo 请先安装 Python 3.11+ 并添加到PATH
        echo ==============================
        echo.
        pause
        exit /b 1
    )
)
set PYTHON_CMD=.venv\Scripts\python.exe
echo [成功] 虚拟环境就绪
echo.

echo [4/6] 启动 Milvus 数据库...
docker --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ==============================
    echo 【错误】Docker 未安装或未启动！
    echo 请先启动 Docker Desktop
    echo ==============================
    echo.
    pause
    exit /b 1
)

docker compose -f vector-database.yml up -d
if errorlevel 1 (
    echo.
    echo ==============================
    echo 【错误】Docker 启动失败
    echo 请检查 vector-database.yml 是否存在
    echo ==============================
    echo.
    pause
    exit /b 1
)
timeout /t 10 /nobreak >nul
echo [成功] Milvus 就绪
echo.

echo [5/6] 启动 CLS MCP 服务...
start "CLS MCP Server" %PYTHON_CMD% mcp_servers/cls_server.py
timeout /t 2 >nul

echo [6/6] 启动 Monitor MCP 服务...
start "Monitor MCP Server" %PYTHON_CMD% mcp_servers/monitor_server.py
timeout /t 2 >nul

echo [7/8] 启动 FastAPI...
start "SuperBizAgent API" %PYTHON_CMD% -m uvicorn app.main:app --host 0.0.0.0 --port 9900
timeout /t 15 >nul

echo.
echo ====================================
echo 服务启动完成！
echo Web 界面: http://localhost:9900
echo ====================================
echo.
pause