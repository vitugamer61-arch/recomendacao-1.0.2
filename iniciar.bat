@echo off
REM ============================================================================
REM iniciar.bat - Ponto de Entrada Unificado do Sistema de Recomendacao Musical
REM ============================================================================
REM Executa todo o fluxo do sistema com duplo clique no Windows:
REM   1. Verifica / inicia o MPD na porta 6600
REM   2. Verifica o ChromaDB e indexa automaticamente faixas novas da biblioteca
REM   3. Agrupa musicas (clusterizacao K-Means automatica ou incremental)
REM   4. Sincroniza playlists por cluster no MPD (.m3u)
REM   5. Inicia o Monitor de Precisao (Dashboard Streamlit) em paralelo
REM   6. Mantem o recomendador ativo em primeiro plano (Modo: cluster)
REM
REM Flags opcionais (passar como argumentos):
REM   --recalcular     Refaz a clusterizacao do zero
REM   --modo <modo>    Modo de clusterizacao: refinado (padrao), nao-supervisionado, por-genero
REM   --sem-cluster    Inicia somente o recomendador K-NN classico
REM   --sem-monitor    Nao inicia o Monitor de Precisao (Streamlit)
REM   --pasta <caminho> Define o caminho da pasta de musicas
REM   --no-gui         Executa o recomendador no console sem janela grafica
REM ============================================================================

cd /d "%~dp0"

set "PYTHON_EXE=%~dp0audio_pipeline\venv\Scripts\python.exe"
set "INICIAR_SCRIPT=%~dp0audio_pipeline\iniciar.py"

if not exist "%PYTHON_EXE%" (
    echo [ERRO] Ambiente virtual Python nao encontrado em:
    echo        %PYTHON_EXE%
    echo Verifique a instalacao do venv em audio_pipeline\venv.
    pause
    exit /b 1
)

if not exist "%INICIAR_SCRIPT%" (
    echo [ERRO] Script iniciar.py nao encontrado em:
    echo        %INICIAR_SCRIPT%
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%INICIAR_SCRIPT%" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [AVISO] O sistema foi encerrado com codigo de erro %ERRORLEVEL%.
    pause
)
