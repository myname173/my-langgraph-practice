# scripts/start_backend.ps1
# 一键启动多媒体 Agent 全链路：即梦 docker + LangGraph Server + 媒体静态服务 + 前端 Vite。
#
# 关键修复：在启动前设置 PYTHONUTF8=1，强制 Python 以 UTF-8 作为
# 默认文件系统/标准流编码，规避两类 Windows GBK 编码问题：
#   1) langgraph_api 读取 openapi 文件时报 UnicodeDecodeError；
#   2) python-dotenv 用系统默认 GBK 读取含中文的 .env 时报
#      UnicodeDecodeError: 'gbk' codec can't decode byte 0x80。
# 注意：必须在本脚本内用 $env:PYTHONUTF8="1" 设置，手动执行
# `python -m langgraph_cli dev` 不带此变量会触发上述第 2 类问题。
#
# 用法（PowerShell）：
#   .\scripts\start_backend.ps1
# 可选参数：
#   -LangGraphPort 2024   # LangGraph Server 端口
#   -MediaPort 8900       # 媒体静态服务端口
#   -JimengPort 8090      # 即梦 jimeng-api docker 端口（与 .env 的 JIMENG_BASE_URL 对齐）
#   -SkipJimengDocker     # 跳过即梦 docker 拉起（用户已手动起好）
#   -SkipHealthCheck      # 跳过即梦干跑健康检查
#
# 干跑健康检查仅探测 GET {base}/v1/models + Bearer 鉴权，
# 绝不调用 generations 端点，避免浪费即梦每日免费积分。

param(
    [int]$LangGraphPort = 2024,
    [int]$MediaPort = 8900,
    [int]$JimengPort = 8090,
    [switch]$SkipJimengDocker = $false,
    [switch]$SkipHealthCheck = $false
)

$ErrorActionPreference = "Stop"

# ---- 编码修复：必须在 python 进程启动前设置 ----
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")

# ─────────────────────────────────────────────────────────────
# 端口就绪轮询等待：确保服务真正监听后再继续，避免“脚本打印完成但
# 服务还没起来 → 用户立刻打开网页/历史会话打不开”。
# ─────────────────────────────────────────────────────────────
function Wait-Port {
    param(
        [int]$Port,
        [string]$Name,
        [int]$TimeoutSec = 120
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($c) {
            Write-Host "  [OK] $Name 已就绪  http://localhost:$Port" -ForegroundColor Green
            return $true
        }
        Start-Sleep -Milliseconds 800
    }
    Write-Host "  [WARN] $Name 未在 ${TimeoutSec}s 内就绪（端口 $Port 未监听），可能仍需手动等待或启动失败" -ForegroundColor Yellow
    return $false
}

Write-Host "[start_backend] PYTHONUTF8=$env:PYTHONUTF8  PYTHONIOENCODING=$env:PYTHONIOENCODING" -ForegroundColor Cyan
Write-Host "[start_backend] ROOT = $ROOT" -ForegroundColor Cyan

# ─────────────────────────────────────────────────────────────
# 0) 解析 .env 中的即梦配置（仅读取，不写入）
# ─────────────────────────────────────────────────────────────
$envPath = Join-Path $ROOT ".env"
$jimengBaseUrl = "http://localhost:$JimengPort/v1"
$jimengSessionId = ""
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $kv = $line -split "=", 2
            if ($kv.Length -eq 2) {
                $key = $kv[0].Trim()
                $val = $kv[1].Trim()
                switch ($key) {
                    "JIMENG_BASE_URL"  { $jimengBaseUrl = $val }
                    "JIMENG_SESSION_ID" { $jimengSessionId = $val }
                }
            }
        }
    }
}
Write-Host "[start_backend] JIMENG_BASE_URL   = $jimengBaseUrl" -ForegroundColor Cyan
if ($jimengSessionId) {
    $mask = if ($jimengSessionId.Length -gt 8) { $jimengSessionId.Substring(0, 4) + "..." + $jimengSessionId.Substring($jimengSessionId.Length - 4) } else { $jimengSessionId }
    Write-Host "[start_backend] JIMENG_SESSION_ID = $mask (len=$($jimengSessionId.Length))" -ForegroundColor Cyan
} else {
    Write-Host "[start_backend] JIMENG_SESSION_ID = <未配置>（即梦相关功能将不可用）" -ForegroundColor Yellow
}

# ─────────────────────────────────────────────────────────────
# 1) 即梦 docker 服务（拉起 / 复用）
# ─────────────────────────────────────────────────────────────
$JIMENG_CONTAINER = "jimeng-free-api-all"
$JIMENG_IMAGE = "vinlic/jimeng-free-api:latest"
$jimengReady = $false

if ($SkipJimengDocker) {
    Write-Host "[start_backend] 已指定 -SkipJimengDocker，跳过 docker 操作。" -ForegroundColor Yellow
} else {
    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCmd) {
        Write-Host "[start_backend] 未检测到 docker CLI，跳过即梦服务拉起。请先安装 Docker Desktop。" -ForegroundColor Yellow
    } else {
        try {
            $dockerInfo = & docker info 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[start_backend] docker daemon 未运行（请启动 Docker Desktop），跳过。" -ForegroundColor Yellow
            } else {
                $existing = (& docker ps -a --filter "name=^${JIMENG_CONTAINER}$" --format "{{.Names}}|{{.Status}}" 2>&1) | Select-Object -First 1
                if ($existing -and $existing -match "^$JIMENG_CONTAINER\|") {
                    $status = ($existing -split "\|")[1]
                    if ($status -match "^Up ") {
                        Write-Host "[start_backend] 即梦容器已运行：$existing" -ForegroundColor Green
                        $jimengReady = $true
                    } else {
                        Write-Host "[start_backend] 即梦容器存在但未运行（$status），执行 docker start ..." -ForegroundColor Yellow
                        & docker start $JIMENG_CONTAINER | Out-Null
                        $jimengReady = $true
                    }
                } else {
                    Write-Host "[start_backend] 拉起即梦容器：$JIMENG_IMAGE （端口 $JimengPort）..." -ForegroundColor Green
                    & docker run -it -d --name $JIMENG_CONTAINER -p "${JimengPort}:8000" -e TZ=Asia/Shanghai $JIMENG_IMAGE
                    if ($LASTEXITCODE -eq 0) {
                        $jimengReady = $true
                        Write-Host "[start_backend] 即梦容器启动成功（首次需拉镜像，可能 10-60s）" -ForegroundColor Green
                    } else {
                        Write-Host "[start_backend] docker run 失败，请检查镜像/端口。" -ForegroundColor Red
                    }
                }
            }
        } catch {
            Write-Host "[start_backend] docker 检测异常：$($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

# ─────────────────────────────────────────────────────────────
# 2) 干跑健康检查（仅 GET /v1/models + Bearer，绝不触发生成）
# ─────────────────────────────────────────────────────────────
$jimengHealth = "unknown"   # ok | auth_fail | unreachable | skipped
$jimengHealthDetail = ""
if ($SkipHealthCheck) {
    $jimengHealth = "skipped"
    $jimengHealthDetail = "用户指定 -SkipHealthCheck"
} elseif (-not $jimengSessionId) {
    $jimengHealth = "unreachable"
    $jimengHealthDetail = "未配置 JIMENG_SESSION_ID，跳过鉴权探活"
} else {
    $firstSid = ($jimengSessionId -split ",")[0].Trim()
    $headers = @{ Authorization = "Bearer $firstSid" }
    try {
        $resp = Invoke-WebRequest -Uri "$jimengBaseUrl/models" -Headers $headers -TimeoutSec 8 -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $jimengHealth = "ok"
            $jimengHealthDetail = "HTTP 200，Bearer 鉴权通过"
        } else {
            $jimengHealth = "auth_fail"
            $jimengHealthDetail = "HTTP $($resp.StatusCode)"
        }
    } catch {
        $msg = $_.Exception.Message
        if ($msg -match "401|403") {
            $jimengHealth = "auth_fail"
        } elseif ($msg -match "404") {
            # 即梦部分实现没有 /v1/models，回退到 /，只验证服务可达
            try {
                $rootResp = Invoke-WebRequest -Uri $jimengBaseUrl.TrimEnd("/v1") -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
                if ($rootResp.StatusCode -ge 200 -and $rootResp.StatusCode -lt 500) {
                    $jimengHealth = "ok"
                    $jimengHealthDetail = "服务可达（/v1/models 不存在，已用根路径兜底 HTTP $($rootResp.StatusCode)）"
                } else {
                    $jimengHealth = "auth_fail"
                    $jimengHealthDetail = "HTTP $($rootResp.StatusCode)"
                }
            } catch {
                $jimengHealth = "unreachable"
                $jimengHealthDetail = $msg
            }
        } else {
            $jimengHealth = "unreachable"
            $jimengHealthDetail = $msg
        }
    }
}

# ─────────────────────────────────────────────────────────────
# 3) 启动 LangGraph Server（端口 $LangGraphPort）
# ─────────────────────────────────────────────────────────────
Write-Host "[start_backend] 启动 LangGraph Server :$LangGraphPort ..." -ForegroundColor Green
Start-Process -WorkingDirectory $ROOT -FilePath "python" `
    -ArgumentList "-m","langgraph_cli","dev","--port",$LangGraphPort `
    -NoNewWindow

# ─────────────────────────────────────────────────────────────
# 4) 媒体静态服务（端口 $MediaPort）
# ─────────────────────────────────────────────────────────────
Write-Host "[start_backend] 启动媒体静态服务 :$MediaPort ..." -ForegroundColor Green
Start-Process -WorkingDirectory $ROOT -FilePath "python" `
    -ArgumentList "-m","src.agent.multimedia.static_server" `
    -Environment @{ MEDIA_SERVER_PORT = "$MediaPort" } `
    -NoNewWindow

# ─────────────────────────────────────────────────────────────
# 5) 前端 Vite（端口 5173）
#   注意：必须用 npm.cmd（而非 npm）启动，否则 PowerShell 7 下
#   Start-Process 无法把 npm.ps1 作为前台服务可靠拉起，Vite 会
#   启动即退，导致 http://localhost:5173 打不开。
# ─────────────────────────────────────────────────────────────
Write-Host "[start_backend] 启动前端 Vite :5173 ..." -ForegroundColor Green
Start-Process -WorkingDirectory (Join-Path $ROOT "frontend") -FilePath "npm.cmd" `
    -ArgumentList "run","dev" `
    -NoNewWindow

# ─────────────────────────────────────────────────────────────
# 5.5) 端口就绪等待：三个核心服务全部监听后再继续。
#      历史会话打不开的根因之一就是服务未真正起来用户就点了网页，
#      这里统一等待 2024 / 8900 / 5173 就绪，从机制上消除该问题。
# ─────────────────────────────────────────────────────────────
Write-Host "[start_backend] 等待服务端口就绪 ..." -ForegroundColor Cyan
$lgReady = Wait-Port -Port $LangGraphPort -Name "LangGraph Server" -TimeoutSec 120
$mediaReady = Wait-Port -Port $MediaPort -Name "媒体静态服务" -TimeoutSec 60
$viteReady = Wait-Port -Port 5173 -Name "前端 Vite" -TimeoutSec 90

# ─────────────────────────────────────────────────────────────
# 6) 打印启动结果 + 全部访问链接
# ─────────────────────────────────────────────────────────────
$jimengStatusColor = switch ($jimengHealth) {
    "ok"          { "Green" }
    "auth_fail"   { "Yellow" }
    "unreachable" { "Red" }
    default       { "DarkGray" }
}
$jimengStatusText = switch ($jimengHealth) {
    "ok"          { "● 健康" }
    "auth_fail"   { "● 鉴权失败" }
    "unreachable" { "● 不可达" }
    "skipped"     { "○ 已跳过" }
    default       { "? 未知" }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  全栈启动完成" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  前端（Vue/React 控制台）  : http://localhost:5173" -ForegroundColor White
Write-Host "  LangGraph Server (API)    : http://127.0.0.1:$LangGraphPort" -ForegroundColor White
Write-Host "  LangGraph Studio (UI)     : https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:$LangGraphPort" -ForegroundColor White
Write-Host "  媒体静态服务（产物）       : http://127.0.0.1:$MediaPort" -ForegroundColor White
Write-Host "  即梦 jimeng-api          : $jimengBaseUrl.TrimEnd("/v1")" -ForegroundColor White
Write-Host "  即梦健康检查              : $jimengStatusText  ($jimengHealthDetail)" -ForegroundColor $jimengStatusColor
Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  健康检查仅 GET /v1/models + Bearer，零积分消耗。" -ForegroundColor DarkGray
Write-Host "  真实生图/生视频请通过前端 '任务启动区' 触发。" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Magenta

if ($jimengHealth -eq "unreachable") {
    Write-Host ""
    Write-Host "  提示：即梦服务尚未就绪，可能是 docker 首次拉镜像中。" -ForegroundColor Yellow
    Write-Host "  等待 30-60s 后访问 $jimengBaseUrl.TrimEnd("/v1") 即可。" -ForegroundColor Yellow
}

# ─────────────────────────────────────────────────────────────
# 7) 自动打开浏览器（仅当前端 Vite 就绪时，避免打开空白页）
# ─────────────────────────────────────────────────────────────
if ($viteReady) {
    Write-Host ""
    Write-Host "[start_backend] 前端已就绪，自动打开 http://localhost:5173 ..." -ForegroundColor Cyan
    Start-Process "http://localhost:5173"
} else {
    Write-Host ""
    Write-Host "[start_backend] 前端未就绪，未自动打开浏览器。请稍候手动访问 http://localhost:5173" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "[start_backend] 请保持此窗口开启，关闭窗口会终止后台服务。" -ForegroundColor DarkGray
