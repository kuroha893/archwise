param(
    [int]$GatewayPort = 8010,
    [int]$ReasoningPort = 8011,
    [int]$KnowledgePort = 8012,
    [int]$LlmGatewayPort = 8013
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogDir = Join-Path $Root "logs"
$PidFile = Join-Path $Root ".archwise-pids.json"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Python = if ($env:PYTHON) { $env:PYTHON } else { "py" }

function Assert-PortAvailable {
    param([int]$Port)
    $listener = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        throw "Port $Port is already in use by process $($listener.OwningProcess)."
    }
}

function Quote-PowerShellValue {
    param([string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

function Start-ArchWiseService {
    param(
        [string]$Name,
        [string]$Module,
        [int]$Port,
        [hashtable]$Environment = @{}
    )

    $commands = @()
    $commands += "[Environment]::SetEnvironmentVariable('PYTHONPATH', $(Quote-PowerShellValue $Root), 'Process')"
    $commands += "[Environment]::SetEnvironmentVariable('ARCHWISE_SERVICE_NAME', $(Quote-PowerShellValue $Name), 'Process')"
    # Each service receives only the environment values it needs for its boundary.
    foreach ($key in $Environment.Keys) {
        $commands += "[Environment]::SetEnvironmentVariable($(Quote-PowerShellValue $key), $(Quote-PowerShellValue ([string]$Environment[$key])), 'Process')"
    }
    $commands += "Set-Location $(Quote-PowerShellValue $Root)"
    $commands += "& $(Quote-PowerShellValue $Python) -m uvicorn ${Module}:app --host 127.0.0.1 --port $Port"
    $command = $commands -join "; "

    $stdout = Join-Path $LogDir "$Name.out.log"
    $stderr = Join-Path $LogDir "$Name.err.log"
    $process = Start-Process powershell `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru

    [pscustomobject]@{
        name = $Name
        module = $Module
        port = $Port
        pid = $process.Id
        stdout = $stdout
        stderr = $stderr
    }
}

foreach ($port in @($GatewayPort, $ReasoningPort, $KnowledgePort, $LlmGatewayPort)) {
    Assert-PortAvailable -Port $port
}

$services = @()
$services += Start-ArchWiseService -Name "llm-gateway" -Module "app.llm_gateway_main" -Port $LlmGatewayPort
$services += Start-ArchWiseService -Name "knowledge" -Module "app.knowledge_main" -Port $KnowledgePort -Environment @{
    LLM_BASE_URL = "http://127.0.0.1:$LlmGatewayPort/v1"
    EMBEDDING_BASE_URL = "http://127.0.0.1:$LlmGatewayPort/v1"
}
$services += Start-ArchWiseService -Name "reasoning" -Module "app.reasoning_main" -Port $ReasoningPort -Environment @{
    KNOWLEDGE_SERVICE_URL = "http://127.0.0.1:$KnowledgePort"
    LLM_BASE_URL = "http://127.0.0.1:$LlmGatewayPort/v1"
    EMBEDDING_BASE_URL = "http://127.0.0.1:$LlmGatewayPort/v1"
}
$services += Start-ArchWiseService -Name "gateway" -Module "app.main" -Port $GatewayPort -Environment @{
    REASONING_SERVICE_URL = "http://127.0.0.1:$ReasoningPort"
    KNOWLEDGE_SERVICE_URL = "http://127.0.0.1:$KnowledgePort"
    LLM_GATEWAY_SERVICE_URL = "http://127.0.0.1:$LlmGatewayPort"
}

$services | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $PidFile -Encoding UTF8
$services | Format-Table -AutoSize
Write-Host "ArchWise microservices started. Gateway: http://127.0.0.1:$GatewayPort"
Write-Host "PID file: $PidFile"
