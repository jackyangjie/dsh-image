import type { Context } from '@deepseek-ai/cordis'
import { defineTool, type ToolRunContext } from '@deepseek-ai/dsh-tools'

import {
  Config, SIZES, RATIOS, VIDEO_RESOLUTIONS, PROVIDER_IDS,
  type Size, type Ratio, type VideoResolution, type ProviderId,
} from './config.js'

export { Config, SIZES, RATIOS, VIDEO_RESOLUTIONS, PROVIDER_IDS }
export type { Size, Ratio, VideoResolution, ProviderId }

export const name = 'dsh-agnes-image'
export const inject = ['tools']

// ─── Provider Registry ────────────────────────────────────────────────

type ProviderFormat = 'openai' | 'google' | 'dashscope' | 'replicate' | 'runway'

interface ProviderMeta {
  id: string
  type: 'image' | 'video'
  format: ProviderFormat
  baseUrl: string
  defaultModel: string
  keyField: keyof Config
}

const PROVIDER_REGISTRY: Record<string, ProviderMeta> = {
  agnes: {
    id: 'agnes', type: 'image', format: 'openai',
    baseUrl: 'https://api.agnes-ai.cn/v1',
    defaultModel: 'agnes-image-2.5-flash',
    keyField: 'agnesApiKey',
  },
  openai: {
    id: 'openai', type: 'image', format: 'openai',
    baseUrl: 'https://api.openai.com/v1',
    defaultModel: 'gpt-image-2',
    keyField: 'openaiApiKey',
  },
  google: {
    id: 'google', type: 'image', format: 'google',
    baseUrl: 'https://generativelanguage.googleapis.com',
    defaultModel: 'gemini-2.0-flash-preview-image-generation',
    keyField: 'googleApiKey',
  },
  dashscope: {
    id: 'dashscope', type: 'image', format: 'dashscope',
    baseUrl: 'https://dashscope.aliyuncs.com',
    defaultModel: 'wanx-v1',
    keyField: 'dashscopeApiKey',
  },
  zai: {
    id: 'zai', type: 'image', format: 'openai',
    baseUrl: 'https://api.z.ai/api/paas/v4',
    defaultModel: 'glm-image',
    keyField: 'zaiApiKey',
  },
  replicate: {
    id: 'replicate', type: 'image', format: 'replicate',
    baseUrl: 'https://api.replicate.com',
    defaultModel: 'google/nano-banana',
    keyField: 'replicateApiKey',
  },
  'google-veo': {
    id: 'google-veo', type: 'video', format: 'google',
    baseUrl: 'https://generativelanguage.googleapis.com',
    defaultModel: 'veo-3.0',
    keyField: 'googleApiKey',
  },
  runway: {
    id: 'runway', type: 'video', format: 'runway',
    baseUrl: 'https://api.runwayml.com',
    defaultModel: 'gen4',
    keyField: 'runwayApiKey',
  },
}

// ─── Normalized Result ────────────────────────────────────────────────

export type GenerationResult = {
  type: string
  provider: string
  model: string
  url: string | null
  b64_json: string | null
  taskId: string | null
  size: string
  ratio: string
  duration: number
  createdAt: number
  error: string | null
}

// ─── Logging ──────────────────────────────────────────────────────────

function getLogger(ctx: unknown) {
  const candidate = (ctx as { logger?: unknown }).logger
  try {
    const logger = typeof candidate === 'function' ? candidate('agnes-image') : candidate
    if (logger && typeof (logger as { info?: unknown }).info === 'function') {
      return logger as Record<string, ((msg: string) => void) | undefined>
    }
  } catch { /* fall through */ }
  return {
    info: (m: string) => console.log(`[agnes-image] ${m}`),
    warn: (m: string) => console.warn(`[agnes-image] ${m}`),
    error: (m: string) => console.error(`[agnes-image] ${m}`),
    debug: undefined,
  }
}

// ─── API Key Lookup ───────────────────────────────────────────────────

function getApiKey(providerMeta: ProviderMeta, config: Config): string {
  return String(config[providerMeta.keyField])
}

function getApiKeyHint(providerMeta: ProviderMeta): string {
  const keyNames: Record<string, string> = {
    agnesApiKey: 'AGNES_API_KEY',
    openaiApiKey: 'OPENAI_API_KEY',
    googleApiKey: 'GOOGLE_API_KEY',
    dashscopeApiKey: 'DASHSCOPE_API_KEY',
    zaiApiKey: 'ZAI_API_KEY (or BIGMODEL_API_KEY)',
    replicateApiKey: 'REPLICATE_API_TOKEN',
    runwayApiKey: 'RUNWAY_API_KEY',
  }
  return keyNames[providerMeta.keyField as string] ?? 'Unknown'
}

// ─── OpenAI-compatible Request Builder ────────────────────────────────

function buildOpenAIBody(params: {
  prompt: string
  model: string
  size?: string
  ratio?: string
  image?: string[]
  returnBase64?: boolean
  responseFormat?: string
}): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: params.model,
    prompt: params.prompt,
  }

  // Size mapping: convert tier to pixel dimensions for OpenAI-compatible APIs
  if (params.size) {
    const sizeMap: Record<string, string> = {
      '1K': '1024x1024', '2K': '1024x768', '3K': '1536x1024', '4K': '1920x1080',
    }
    // Use ratio-aware dimensions
    if (params.ratio) {
      const ratioMap: Record<string, Record<string, string>> = {
        '1:1': { '1K': '1024x1024', '2K': '2048x2048', '3K': '3072x3072', '4K': '4096x4096' },
        '16:9': { '1K': '1312x736', '2K': '2624x1472', '3K': '3936x2208', '4K': '5248x2944' },
        '9:16': { '1K': '736x1312', '2K': '1472x2624', '3K': '2208x3936', '4K': '2944x5248' },
        '4:3': { '1K': '1152x864', '2K': '2304x1728', '3K': '3456x2592', '4K': '4608x3456' },
        '3:4': { '1K': '864x1152', '2K': '1728x2304', '3K': '2592x3456', '4K': '3456x4608' },
      }
      const ratioSizes = ratioMap[params.ratio]
      if (ratioSizes && ratioSizes[params.size]) {
        body.size = ratioSizes[params.size]
      } else {
        body.size = sizeMap[params.size] ?? params.size
      }
    } else {
      body.size = sizeMap[params.size] ?? params.size
    }
  }

  // extra_body for response format and images
  const extraBody: Record<string, unknown> = {}
  if (params.responseFormat) extraBody.response_format = params.responseFormat
  if (params.image && params.image.length > 0) extraBody.image = params.image
  if (params.returnBase64) body.return_base64 = true
  if (Object.keys(extraBody).length > 0) body.extra_body = extraBody

  return body
}

// ─── Google Request Builder ───────────────────────────────────────────

function buildGoogleBody(params: {
  prompt: string
  ratio?: string
}): Record<string, unknown> {
  const body: Record<string, unknown> = {
    contents: [{ parts: [{ text: params.prompt }] }],
  }
  if (params.ratio) {
    body.generationConfig = {
      responseModalities: ['IMAGE'],
    }
  }
  return body
}

// ─── DashScope Request Builder ────────────────────────────────────────

function buildDashScopeBody(params: {
  prompt: string
  model: string
  size?: string
}): Record<string, unknown> {
  const sizeMap: Record<string, string> = {
    '1K': '1024*1024', '2K': '1440*1440', '3K': '1600*1600', '4K': '2048*2048',
  }
  return {
    model: params.model,
    input: { prompt: params.prompt },
    parameters: {
      size: sizeMap[params.size ?? '1K'] ?? '1024*1024',
    },
  }
}

// ─── Replicate Request Builder ────────────────────────────────────────

function buildReplicateBody(params: {
  prompt: string
  ratio?: string
  image?: string[]
}): Record<string, unknown> {
  const input: Record<string, unknown> = { prompt: params.prompt }
  if (params.ratio) input.aspect_ratio = params.ratio
  if (params.image && params.image.length > 0) input.reference_image = params.image[0]
  return { input }
}

// ─── Video Request Builders ───────────────────────────────────────────

function buildGoogleVeoBody(params: {
  prompt: string
  duration?: number
  ratio?: string
  resolution?: string
}): Record<string, unknown> {
  const parameters: Record<string, unknown> = { sampleCount: 1 }
  if (params.duration) parameters.duration = `${params.duration}s`
  if (params.ratio) parameters.aspectRatio = params.ratio
  return {
    instances: [{ prompt: params.prompt }],
    parameters,
  }
}

function buildRunwayBody(params: {
  prompt: string
  duration?: number
  ratio?: string
}): Record<string, unknown> {
  return {
    prompt: params.prompt,
    model: 'gen4',
    duration: params.duration ?? 8,
    ...(params.ratio ? { aspectRatio: params.ratio } : {}),
  }
}

// ─── Video Polling Helpers ────────────────────────────────────────────

const POLL_INTERVAL_MS = 5000

async function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const id = setTimeout(resolve, ms)
    signal.addEventListener('abort', () => {
      clearTimeout(id)
      reject(new Error('aborted'))
    }, { once: true })
  })
}

async function pollGoogleVeo(
  baseUrl: string, apiKey: string, operationName: string,
  signal: AbortSignal, timeoutMs: number,
): Promise<string | null> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    signal.throwIfAborted()
    const url = `${baseUrl}/v1beta/${operationName}`
    const resp = await fetch(url, {
      headers: { 'x-goog-api-key': apiKey },
      signal,
    })
    if (!resp.ok) {
      throw new Error(`Google Veo polling error (${resp.status})`)
    }
    const data = (await resp.json()) as {
      done: boolean
      error?: { message: string }
      response?: {
        generateVideoResponse?: {
          generatedVideos?: Array<{ uri: string; video?: { mimeType: string; data: string } }>
        }
      }
    }
    if (data.error) throw new Error(`Google Veo error: ${data.error.message}`)
    if (data.done) {
      const videos = data.response?.generateVideoResponse?.generatedVideos
      if (videos && videos.length > 0) {
        const v = videos[0]
        if (v.uri) return v.uri
        if (v.video?.data) return `data:video/mp4;base64,${v.video.data}`
      }
      throw new Error('Google Veo completed but returned no video')
    }
    await sleep(POLL_INTERVAL_MS, signal)
  }
  throw new Error(`Google Veo polling timeout after ${timeoutMs}ms`)
}

async function pollReplicate(
  baseUrl: string, apiKey: string, predictionId: string,
  signal: AbortSignal, timeoutMs: number,
): Promise<string | null> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    signal.throwIfAborted()
    const resp = await fetch(`${baseUrl}/v1/predictions/${predictionId}`, {
      headers: { Authorization: `Token ${apiKey}` },
      signal,
    })
    if (!resp.ok) throw new Error(`Replicate polling error (${resp.status})`)
    const data = (await resp.json()) as {
      status: string
      output?: string | string[]
      error?: string
    }
    if (data.error) throw new Error(`Replicate error: ${data.error}`)
    if (data.status === 'succeeded') {
      if (Array.isArray(data.output)) return data.output[0] ?? null
      return typeof data.output === 'string' ? data.output : null
    }
    if (data.status === 'failed' || data.status === 'canceled') {
      throw new Error(`Replicate ${data.status}: ${data.error ?? 'unknown error'}`)
    }
    await sleep(POLL_INTERVAL_MS, signal)
  }
  throw new Error(`Replicate polling timeout after ${timeoutMs}ms`)
}

async function pollRunway(
  baseUrl: string, apiKey: string, videoId: string,
  signal: AbortSignal, timeoutMs: number,
): Promise<string | null> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    signal.throwIfAborted()
    const resp = await fetch(`${baseUrl}/v1/videos/${videoId}`, {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal,
    })
    if (!resp.ok) throw new Error(`Runway polling error (${resp.status})`)
    const data = (await resp.json()) as {
      status: string
      output?: string[]
      errorMessage?: string
    }
    if (data.errorMessage) throw new Error(`Runway error: ${data.errorMessage}`)
    if (data.status === 'completed') {
      return data.output?.[0] ?? null
    }
    if (data.status === 'failed') {
      throw new Error(`Runway failed: ${data.errorMessage ?? 'unknown error'}`)
    }
    await sleep(POLL_INTERVAL_MS, signal)
  }
  throw new Error(`Runway polling timeout after ${timeoutMs}ms`)
}

async function pollDashScope(
  baseUrl: string, apiKey: string, taskId: string,
  signal: AbortSignal, timeoutMs: number,
): Promise<string | null> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    signal.throwIfAborted()
    const resp = await fetch(`${baseUrl}/api/v1/tasks/${taskId}`, {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal,
    })
    if (!resp.ok) throw new Error(`DashScope polling error (${resp.status})`)
    const data = (await resp.json()) as {
      output: {
        task_status: string
        results?: Array<{ url: string }>
        message?: string
      }
    }
    if (data.output.task_status === 'SUCCEEDED') {
      return data.output.results?.[0]?.url ?? null
    }
    if (data.output.task_status === 'FAILED') {
      throw new Error(`DashScope failed: ${data.output.message ?? 'unknown error'}`)
    }
    await sleep(POLL_INTERVAL_MS, signal)
  }
  throw new Error(`DashScope polling timeout after ${timeoutMs}ms`)
}

// ─── Main Apply ───────────────────────────────────────────────────────

export function apply(ctx: Context, config: Config) {
  const log = getLogger(ctx)

  const inflight = new Set<AbortController>()
  ctx.effect(() => () => {
    for (const controller of inflight) controller.abort()
    inflight.clear()
  }, 'agnes-image:inflight')

  ctx.tools.register(
    defineTool({
      name: 'agnes_image_gen',
      description:
        '多模型图片/视频生成工具。支持 8 个 Provider：Agnes、OpenAI、Google、DashScope、Z.AI、Replicate（图片）+ Google Veo、Runway（视频）。' +
        '当用户要求生成图片、创建图片、绘制图片、做图像编辑/风格转换、生成视频时使用此工具。',

      parameters: {
        prompt: {
          type: 'string',
          required: true,
          description: '图片描述或视频生成指令。推荐结构：主体 + 场景 + 风格 + 光照 + 构图 + 质量要求。',
        },
        provider: {
          type: 'string',
          enum: PROVIDER_IDS,
          description: '选择 Provider。图片: agnes|openai|google|dashscope|zai|replicate。视频: google-veo|runway。默认从配置读取。',
        },
        model: {
          type: 'string',
          description: '覆盖默认模型名称。例如 replicate 下可用 "black-forest-labs/flux-1-schnell"。',
        },
        size: {
          type: 'string',
          enum: SIZES,
          description: '图片输出分辨率档位（1K/2K/3K/4K），默认 2K。',
        },
        ratio: {
          type: 'string',
          enum: RATIOS,
          description: '输出宽高比，默认 16:9。',
        },
        image: {
          type: 'array',
          items: { type: 'string' },
          description: '输入图片数组（图生图）。支持 URL 或 Data URI Base64。',
        },
        duration: {
          type: 'number',
          description: '视频时长（秒），默认 8。仅对视频 Provider 有效。',
        },
        resolution: {
          type: 'string',
          enum: VIDEO_RESOLUTIONS,
          description: '视频分辨率（720p/1080p）。仅对视频 Provider 有效。',
        },
        return_base64: {
          type: 'boolean',
          description: '是否以 Base64 返回图片数据。仅对图片 Provider 有效。',
        },
        response_format: {
          type: 'string',
          enum: ['url', 'b64_json'],
          description: '输出格式：url 或 b64_json。',
        },
      },

      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            type: { type: 'string', required: true },
            provider: { type: 'string', required: true },
            model: { type: 'string', required: true },
            url: { oneOf: [{ type: 'string' }, { type: 'null' }], required: true },
            b64_json: { oneOf: [{ type: 'string' }, { type: 'null' }], required: true },
            taskId: { oneOf: [{ type: 'string' }, { type: 'null' }], required: true },
            size: { type: 'string', required: true },
            ratio: { type: 'string', required: true },
            duration: { type: 'number', required: true },
            createdAt: { type: 'number', required: true },
            error: { oneOf: [{ type: 'string' }, { type: 'null' }], required: true },
          },
        },
        render: (_args, value) => {
          const blocks: Array<{ type: 'text'; text: string }> = []

          if (value.error) {
            blocks.push({
              type: 'text',
              text: `❌ 生成失败\n\nProvider: ${value.provider} | Model: ${value.model}\n错误: ${value.error}`,
            })
            return blocks
          }

          if (value.type === 'image') {
            if (value.url) {
              blocks.push({
                type: 'text',
                text: `🖼️ 图片已生成\n\n🔗 URL: ${value.url}\n📐 尺寸: ${value.size} (${value.ratio})\n🏷️ Provider: ${value.provider} | Model: ${value.model}`,
              })
            } else if (value.b64_json) {
              blocks.push({
                type: 'text',
                text: `🖼️ 图片已生成（Base64 格式）\n\n📐 尺寸: ${value.size} (${value.ratio})\n📦 数据长度: ${value.b64_json.length} 字符\n🏷️ Provider: ${value.provider} | Model: ${value.model}`,
              })
            } else if (value.taskId) {
              blocks.push({
                type: 'text',
                text: `⏳ 图片生成任务已提交\n\n🆔 Task ID: ${value.taskId}\n🏷️ Provider: ${value.provider} | Model: ${value.model}`,
              })
            }
          } else if (value.type === 'video') {
            if (value.url) {
              blocks.push({
                type: 'text',
                text: `🎬 视频已生成\n\n🔗 URL: ${value.url}\n⏱️ 时长: ${value.duration}s\n📺 分辨率: ${value.size}\n🏷️ Provider: ${value.provider} | Model: ${value.model}`,
              })
            } else if (value.taskId) {
              blocks.push({
                type: 'text',
                text: `⏳ 视频生成任务已提交\n\n🆔 Task ID: ${value.taskId}\n⏱️ 时长: ${value.duration}s\n🏷️ Provider: ${value.provider} | Model: ${value.model}`,
              })
            }
          }

          return blocks
        },
      },

      isConcurrencySafe: () => false,

      async execute(args, exec: ToolRunContext): Promise<GenerationResult> {
        if (exec.signal.aborted) throw new Error('aborted before start')

        const providerId = (args.provider ?? config.defaultProvider) as string
        const providerMeta = PROVIDER_REGISTRY[providerId]

        if (!providerMeta) {
          return {
            type: 'image', provider: providerId, model: '', url: null, b64_json: null,
            taskId: null, size: '', ratio: '', duration: 0, createdAt: Date.now(),
            error: `Unknown provider "${providerId}". Available: ${Object.keys(PROVIDER_REGISTRY).join(', ')}`,
          }
        }

        const apiKey = getApiKey(providerMeta, config)
        if (!apiKey) {
          return {
            type: providerMeta.type, provider: providerId, model: providerMeta.defaultModel,
            url: null, b64_json: null, taskId: null, size: '', ratio: '', duration: 0,
            createdAt: Date.now(),
            error: `Provider "${providerId}" 未配置 API Key。请设置环境变量 ${getApiKeyHint(providerMeta)} 或在插件配置中添加 ${providerMeta.keyField}。`,
          }
        }

        const model = args.model ?? providerMeta.defaultModel
        const isVideo = providerMeta.type === 'video'
        const timeoutMs = isVideo ? config.videoTimeoutMs : config.timeoutMs

        // Create abort controller
        const controller = new AbortController()
        inflight.add(controller)
        const timeoutId = setTimeout(
          () => controller.abort(new Error(`timeout after ${timeoutMs}ms`)),
          timeoutMs,
        )
        const onAbort = () => controller.abort(exec.signal.reason)
        exec.signal.addEventListener('abort', onAbort, { once: true })

        try {
          log.debug?.(`${providerId}: ${isVideo ? 'video' : 'image'} generation start`)

          let result: GenerationResult

          switch (providerMeta.format) {
            // ── OpenAI-compatible (agnes, openai, zai) ──
            case 'openai': {
              const body = buildOpenAIBody({
                prompt: args.prompt, model,
                size: args.size ?? '2K',
                ratio: args.ratio ?? '16:9',
                image: args.image,
                returnBase64: args.return_base64,
                responseFormat: args.response_format,
              })

              const endpoint = `${providerMeta.baseUrl}/images/generations`
              const resp = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
                body: JSON.stringify(body),
                signal: controller.signal,
              })
              if (!resp.ok) throw new Error(`API error (${resp.status}): ${await resp.text()}`)
              const data = (await resp.json()) as {
                created: number
                data: Array<{ url: string | null; b64_json: string | null; revised_prompt: string | null }>
              }
              const img = data.data?.[0]
              result = {
                type: 'image', provider: providerId, model,
                url: img?.url ?? null, b64_json: img?.b64_json ?? null,
                taskId: null, size: args.size ?? '2K', ratio: args.ratio ?? '16:9',
                duration: 0, createdAt: data.created ?? Date.now(), error: null,
              }
              break
            }

            // ── Google (image + veo video) ──
            case 'google': {
              if (providerMeta.type === 'image') {
                // Google image generation
                const body = buildGoogleBody({ prompt: args.prompt, ratio: args.ratio })
                const endpoint = `${providerMeta.baseUrl}/v1beta/models/${model}:generateContent`
                const resp = await fetch(endpoint, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
                  body: JSON.stringify(body),
                  signal: controller.signal,
                })
                if (!resp.ok) throw new Error(`Google API error (${resp.status}): ${await resp.text()}`)
                const data = (await resp.json()) as {
                  candidates?: Array<{ content?: { parts?: Array<{ inlineData?: { mimeType: string; data: string } }> } }>
                }
                const part = data.candidates?.[0]?.content?.parts?.[0]
                result = {
                  type: 'image', provider: providerId, model,
                  url: null, b64_json: part?.inlineData?.data ?? null,
                  taskId: null, size: args.size ?? '1K', ratio: args.ratio ?? '1:1',
                  duration: 0, createdAt: Date.now(), error: null,
                }
              } else {
                // Google Veo video generation (async)
                const body = buildGoogleVeoBody({
                  prompt: args.prompt,
                  duration: args.duration,
                  ratio: args.ratio ?? '16:9',
                  resolution: args.resolution,
                })
                const endpoint = `${providerMeta.baseUrl}/v1beta/models/${model}:predictLongRunning`
                const resp = await fetch(endpoint, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
                  body: JSON.stringify(body),
                  signal: controller.signal,
                })
                if (!resp.ok) throw new Error(`Google Veo API error (${resp.status}): ${await resp.text()}`)
                const submitData = (await resp.json()) as { name: string }
                const operationName = submitData.name
                if (!operationName) throw new Error('Google Veo returned no operation name')

                // Poll for completion
                const videoUrl = await pollGoogleVeo(
                  providerMeta.baseUrl, apiKey, operationName,
                  controller.signal, timeoutMs,
                )
                result = {
                  type: 'video', provider: providerId, model,
                  url: videoUrl, b64_json: null, taskId: null,
                  size: args.resolution ?? '1080p', ratio: args.ratio ?? '16:9',
                  duration: args.duration ?? 8, createdAt: Date.now(), error: null,
                }
              }
              break
            }

            // ── DashScope ──
            case 'dashscope': {
              const body = buildDashScopeBody({ prompt: args.prompt, model, size: args.size })
              const endpoint = `${providerMeta.baseUrl}/api/v1/services/aigc/text2image/image-synthesis`
              const resp = await fetch(endpoint, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  Authorization: `Bearer ${apiKey}`,
                  'X-DashScope-Async': 'enable',
                },
                body: JSON.stringify(body),
                signal: controller.signal,
              })
              if (!resp.ok) throw new Error(`DashScope error (${resp.status}): ${await resp.text()}`)
              const submitData = (await resp.json()) as { output: { task_id: string } }
              const taskId = submitData.output?.task_id
              if (!taskId) throw new Error('DashScope returned no task_id')

              // Poll for completion
              const imageUrl = await pollDashScope(
                providerMeta.baseUrl, apiKey, taskId,
                controller.signal, timeoutMs,
              )
              result = {
                type: 'image', provider: providerId, model,
                url: imageUrl, b64_json: null, taskId: null,
                size: args.size ?? '1K', ratio: args.ratio ?? '1:1',
                duration: 0, createdAt: Date.now(), error: null,
              }
              break
            }

            // ── Replicate ──
            case 'replicate': {
              const body = buildReplicateBody({
                prompt: args.prompt, ratio: args.ratio, image: args.image,
              })
              const endpoint = `${providerMeta.baseUrl}/v1/models/${model}/predictions`
              const resp = await fetch(endpoint, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  Authorization: `Token ${apiKey}`,
                },
                body: JSON.stringify(body),
                signal: controller.signal,
              })
              if (!resp.ok) throw new Error(`Replicate error (${resp.status}): ${await resp.text()}`)
              const submitData = (await resp.json()) as { id: string }
              const predictionId = submitData.id
              if (!predictionId) throw new Error('Replicate returned no prediction ID')

              // Poll for completion
              const outputUrl = await pollReplicate(
                providerMeta.baseUrl, apiKey, predictionId,
                controller.signal, timeoutMs,
              )
              result = {
                type: providerMeta.type, provider: providerId, model,
                url: outputUrl, b64_json: null, taskId: null,
                size: args.size ?? '1K', ratio: args.ratio ?? '16:9',
                duration: args.duration ?? 0, createdAt: Date.now(), error: null,
              }
              break
            }

            // ── Runway ──
            case 'runway': {
              const body = buildRunwayBody({
                prompt: args.prompt,
                duration: args.duration,
                ratio: args.ratio ?? '16:9',
              })
              const endpoint = `${providerMeta.baseUrl}/v1/videos`
              const resp = await fetch(endpoint, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  Authorization: `Bearer ${apiKey}`,
                },
                body: JSON.stringify(body),
                signal: controller.signal,
              })
              if (!resp.ok) throw new Error(`Runway error (${resp.status}): ${await resp.text()}`)
              const submitData = (await resp.json()) as { id: string }
              const videoId = submitData.id
              if (!videoId) throw new Error('Runway returned no video ID')

              // Poll for completion
              const videoUrl = await pollRunway(
                providerMeta.baseUrl, apiKey, videoId,
                controller.signal, timeoutMs,
              )
              result = {
                type: 'video', provider: providerId, model,
                url: videoUrl, b64_json: null, taskId: null,
                size: args.resolution ?? '1080p', ratio: args.ratio ?? '16:9',
                duration: args.duration ?? 8, createdAt: Date.now(), error: null,
              }
              break
            }

            default:
              throw new Error(`Unknown format: ${(providerMeta as { format: string }).format}`)
          }

          log.info?.(`${providerId}: ${result.type} generated OK (model: ${result.model})`)
          return result
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : String(err)
          log.error?.(`${providerId}: ${message}`)
          return {
            type: providerMeta.type, provider: providerId, model,
            url: null, b64_json: null, taskId: null,
            size: '', ratio: '', duration: 0, createdAt: Date.now(),
            error: message,
          }
        } finally {
          clearTimeout(timeoutId)
          exec.signal.removeEventListener('abort', onAbort)
          inflight.delete(controller)
        }
      },
    }),
  )

  // Log which providers have keys configured
  const configured: string[] = []
  for (const [id, meta] of Object.entries(PROVIDER_REGISTRY)) {
    if (getApiKey(meta, config)) configured.push(id)
  }
  log.info?.(`dsh-agnes-image ready. Configured providers: ${configured.length > 0 ? configured.join(', ') : 'NONE — set API keys to enable'}`)
}
