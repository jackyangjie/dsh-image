import Schema from '@deepseek-ai/schemastery'

/** All supported provider IDs */
export const PROVIDER_IDS = [
  'agnes', 'openai', 'google', 'dashscope', 'zai', 'replicate',
  'google-veo', 'runway',
] as const
export type ProviderId = (typeof PROVIDER_IDS)[number]

/** Supported size tiers for image generation */
export const SIZES = ['1K', '2K', '3K', '4K'] as const
export type Size = (typeof SIZES)[number]

/** Supported aspect ratios */
export const RATIOS = [
  '1:1', '3:4', '4:3', '16:9', '9:16', '2:3', '3:2', '21:9',
] as const
export type Ratio = (typeof RATIOS)[number]

/** Video resolutions */
export const VIDEO_RESOLUTIONS = ['720p', '1080p'] as const
export type VideoResolution = (typeof VIDEO_RESOLUTIONS)[number]

/**
 * Config is exported TWICE:
 *   - a TypeScript interface for apply(ctx, config: Config)
 *   - a runtime Schemastery schema for Cordis validation
 */
export interface Config {
  /** Default provider to use when not specified */
  defaultProvider: string
  agnesApiKey: string
  openaiApiKey: string
  googleApiKey: string
  dashscopeApiKey: string
  zaiApiKey: string
  replicateApiKey: string
  runwayApiKey: string
  timeoutMs: number
  videoTimeoutMs: number
}

export const Config: Schema<Config> = Schema.object({
  defaultProvider: Schema.string()
    .default('agnes')
    .description('Default provider for image/video generation.'),

  agnesApiKey: Schema.string()
    .default(process.env.AGNES_API_KEY ?? '')
    .description('Agnes AI API key.'),
  openaiApiKey: Schema.string()
    .default(process.env.OPENAI_API_KEY ?? '')
    .description('OpenAI API key.'),
  googleApiKey: Schema.string()
    .default(process.env.GOOGLE_API_KEY ?? '')
    .description('Google API key (for google and google-veo).'),
  dashscopeApiKey: Schema.string()
    .default(process.env.DASHSCOPE_API_KEY ?? '')
    .description('Alibaba DashScope API key.'),
  zaiApiKey: Schema.string()
    .default(process.env.ZAI_API_KEY ?? process.env.BIGMODEL_API_KEY ?? '')
    .description('Z.AI API key.'),
  replicateApiKey: Schema.string()
    .default(process.env.REPLICATE_API_TOKEN ?? '')
    .description('Replicate API token.'),
  runwayApiKey: Schema.string()
    .default(process.env.RUNWAY_API_KEY ?? '')
    .description('Runway API key.'),

  timeoutMs: Schema.number()
    .min(5000)
    .default(300_000)
    .description('Image generation request timeout (ms).'),
  videoTimeoutMs: Schema.number()
    .min(30_000)
    .default(600_000)
    .description('Video generation timeout including polling (ms).'),
})
