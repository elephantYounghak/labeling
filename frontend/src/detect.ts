import { boxPoints, clamp01, type Mark, type MarkPoint } from './marks'
import type { LabelTool } from './projects'

export type DetectRequest = {
  model: string
  name: string
  image: string
  path: string
  tool: LabelTool
  words: string[]
  word?: string
  x?: number
  y?: number
  width?: number
  height?: number
}

type RawBox = {
  x1: number
  y1: number
  x2: number
  y2: number
  name: string
}

function numberOf(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function nameOf(value: unknown) {
  return typeof value === 'string' ? value.trim() : ''
}

function readPoint(value: unknown): MarkPoint | null {
  if (!value || typeof value !== 'object') return null
  const point = value as { x?: unknown; y?: unknown }
  const x = numberOf(point.x)
  const y = numberOf(point.y)
  if (x === null || y === null) return null
  return { x, y }
}

function readBoxes(value: unknown): RawBox[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!item || typeof item !== 'object') return []
    const row = item as { x1?: unknown; y1?: unknown; x2?: unknown; y2?: unknown; name?: unknown }
    const x1 = numberOf(row.x1)
    const y1 = numberOf(row.y1)
    const x2 = numberOf(row.x2)
    const y2 = numberOf(row.y2)
    if (x1 === null || y1 === null || x2 === null || y2 === null) return []
    return [{ x1, y1, x2, y2, name: nameOf(row.name) }]
  })
}

function readMasks(value: unknown): { points: MarkPoint[]; name: string }[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!item || typeof item !== 'object') return []
    const row = item as { points?: unknown; name?: unknown }
    if (!Array.isArray(row.points)) return []
    const points = row.points.flatMap((point) => {
      const next = readPoint(point)
      return next ? [next] : []
    })
    if (points.length < 3) return []
    return [{ points, name: nameOf(row.name) }]
  })
}

function pixelSpace(values: number[]) {
  return values.some((value) => Math.abs(value) > 1.001)
}

function unit(value: number, size: number, pixel: boolean) {
  if (!pixel) return clamp01(value)
  if (size <= 0) return 0
  return clamp01(value / size)
}

function boxToMark(box: RawBox, natural: { width: number; height: number }, name: string, color: string): Mark | null {
  const pixel = pixelSpace([box.x1, box.y1, box.x2, box.y2])
  const x1 = unit(Math.min(box.x1, box.x2), natural.width, pixel)
  const y1 = unit(Math.min(box.y1, box.y2), natural.height, pixel)
  const x2 = unit(Math.max(box.x1, box.x2), natural.width, pixel)
  const y2 = unit(Math.max(box.y1, box.y2), natural.height, pixel)
  if (x2 - x1 <= 0 || y2 - y1 <= 0) return null
  return {
    id: crypto.randomUUID(),
    tool: 'bbox',
    name,
    color,
    points: boxPoints({ x: x1, y: y1, w: x2 - x1, h: y2 - y1 }),
  }
}

export function detectionsToMarks(
  data: unknown,
  tool: LabelTool,
  natural: { width: number; height: number },
  fallbackName: string,
  colorOf: (name: string) => string,
): Mark[] {
  if (!data || typeof data !== 'object') return []
  const row = data as { boxes?: unknown; masks?: unknown }
  const boxes = readBoxes(row.boxes)
  const masks = readMasks(row.masks)
  const marks: Mark[] = []

  if ((tool === 'polygon' || tool === 'brush') && masks.length > 0) {
    for (const mask of masks) {
      const pixel = pixelSpace(mask.points.flatMap((point) => [point.x, point.y]))
      const name = mask.name || fallbackName
      marks.push({
        id: crypto.randomUUID(),
        tool: 'polygon',
        name,
        color: colorOf(name),
        points: mask.points.map((point) => ({
          x: unit(point.x, natural.width, pixel),
          y: unit(point.y, natural.height, pixel),
        })),
      })
    }
    return marks
  }

  const shaped = boxes.length > 0
    ? boxes
    : masks.map((mask) => {
        const pixel = pixelSpace(mask.points.flatMap((point) => [point.x, point.y]))
        const xs = mask.points.map((point) => unit(point.x, natural.width, pixel))
        const ys = mask.points.map((point) => unit(point.y, natural.height, pixel))
        return {
          x1: Math.min(...xs),
          y1: Math.min(...ys),
          x2: Math.max(...xs),
          y2: Math.max(...ys),
          name: mask.name,
        }
      })

  for (const box of shaped) {
    const name = box.name || fallbackName
    const mark = boxToMark(box, natural, name, colorOf(name))
    if (mark) marks.push(mark)
  }
  return marks
}

export async function detectApi(body: DetectRequest, file: File): Promise<unknown> {
  const form = new FormData()
  form.append('model', body.model)
  form.append('name', body.name)
  form.append('path', body.path)
  form.append('tool', body.tool)
  form.append('words', JSON.stringify(body.words))
  if (body.word) form.append('word', body.word)
  if (body.x !== undefined) form.append('x', String(body.x))
  if (body.y !== undefined) form.append('y', String(body.y))
  if (body.width !== undefined) form.append('width', String(body.width))
  if (body.height !== undefined) form.append('height', String(body.height))
  form.append('image', file, file.name)
  const res = await fetch('/ai/detect', {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    let message = '검출 API가 실패했습니다.'
    try {
      const body = await res.json() as { error?: unknown }
      if (typeof body.error === 'string' && body.error.trim()) {
        message = body.error.trim()
      }
    } catch {
    }
    throw new Error(message)
  }
  return res.json()
}
