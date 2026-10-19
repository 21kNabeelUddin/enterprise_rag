<#
.SYNOPSIS
  Serve the local Qwen3.5-9B (GGUF) as an OpenAI-compatible API via llama.cpp.

.DESCRIPTION
  Runs tools/llama.cpp/llama-server.exe against models/Qwen_Qwen3.5-9B-Q4_K_M.gguf
  and exposes http://127.0.0.1:8000/v1 (OpenAI-compatible) for the query pipeline.

  Tuned for an 8 GB laptop GPU:
    * -ngl 99        all layers on the GPU
    * --flash-attn   flash attention on
    * -ctk/-ctv q8_0 8-bit KV cache (saves VRAM so 8k context fits)
    * -np 1          a single server slot
    * --reasoning off  Qwen "thinking" disabled by default (faster, grounded
                       answers). Pass -Thinking or send
                       chat_template_kwargs={enable_thinking:true} per request
                       to turn it back on.

.EXAMPLE
  .\scripts\serve_llm.ps1
  .\scripts\serve_llm.ps1 -Ctx 4096            # smaller context if VRAM is tight
  .\scripts\serve_llm.ps1 -Thinking            # enable the reasoning mode
  .\scripts\serve_llm.ps1 -Port 8080
#>
[CmdletBinding()]
param(
    [string]$Model = "models\Qwen_Qwen3.5-9B-Q4_K_M.gguf",
    [int]$Port = 8000,
    [int]$Ctx = 8192,
    [int]$GpuLayers = 99,
    [string]$KvType = "q8_0",
    [switch]$Thinking   # enable Qwen thinking mode (default: off)
)

$ErrorActionPreference = "Stop"

$root      = Split-Path -Parent $PSScriptRoot
$server    = Join-Path $root "tools\llama.cpp\llama-server.exe"
$modelPath = if ([System.IO.Path]::IsPathRooted($Model)) { $Model } else { Join-Path $root $Model }

if (-not (Test-Path $server))    { throw "llama-server not found: $server" }
if (-not (Test-Path $modelPath)) { throw "model not found: $modelPath" }

$reasoning = if ($Thinking) { "on" } else { "off" }

Write-Host ""
Write-Host "  llama-server : $server"
Write-Host "  model        : $modelPath"
Write-Host "  endpoint     : http://127.0.0.1:$Port/v1   (OpenAI-compatible)"
Write-Host "  ctx / kv     : $Ctx tokens / $KvType  |  gpu-layers: $GpuLayers  |  thinking: $reasoning"
Write-Host "  stop         : Ctrl+C"
Write-Host ""

& $server `
    -m $modelPath `
    -ngl $GpuLayers `
    -c $Ctx `
    --flash-attn on `
    -ctk $KvType `
    -ctv $KvType `
    -np 1 `
    --reasoning $reasoning `
    --host 127.0.0.1 `
    --port $Port `
    --jinja
