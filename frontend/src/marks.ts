import type { LabelTool } from './projects'

export type MarkPoint = { x: number; y: number }

export type Mark = {
  id: string
  tool: LabelTool
  name: string
  color: string
  points: MarkPoint[]
}

function storageKey(projectId: string) {
  return `labeling.marks.${projectId}`
}

function readMark(value: unknown): Mark | null {
  if (!value || typeof value !== 'object') return null
  const row = value as Partial<Mark>
  if (row.tool !== 'bbox' && row.tool !== 'polygon' && row.tool !== 'brush') return null
  if (typeof row.name !== 'string' || typeof row.color !== 'string' || !Array.isArray(row.points)) return null
  const points = row.points.flatMap((point) => {
    if (!point || typeof point !== 'object') return []
    const item = point as Partial<MarkPoint>
    if (typeof item.x !== 'number' || typeof item.y !== 'number') return []
    return [{ x: item.x, y: item.y }]
  })
  if (points.length === 0) return null
  return { id: typeof row.id === 'string' ? row.id : crypto.randomUUID(), tool: row.tool, name: row.name, color: row.color, points }
}

export function loadMarks(projectId: string, imageName: string): Mark[] {
  try {
    const raw = localStorage.getItem(storageKey(projectId))
    if (!raw) return []
    const all = JSON.parse(raw) as Record<string, unknown>
    const list = all[imageName]
    if (!Array.isArray(list)) return []
    return list.flatMap((item) => {
      const mark = readMark(item)
      return mark ? [mark] : []
    })
  } catch {
    return []
  }
}

export function saveMarks(projectId: string, imageName: string, marks: Mark[]) {
  const raw = localStorage.getItem(storageKey(projectId))
  const all = raw ? (JSON.parse(raw) as Record<string, Mark[]>) : {}
  all[imageName] = marks
  localStorage.setItem(storageKey(projectId), JSON.stringify(all))
}

export function pointNear(a: MarkPoint, b: MarkPoint, gap = 0.03) {
  return Math.hypot(a.x - b.x, a.y - b.y) <= gap
}

export function hitMark(marks: Mark[], point: MarkPoint): Mark | null {
  for (let index = marks.length - 1; index >= 0; index -= 1) {
    if (contains(marks[index], point)) return marks[index]
  }
  return null
}

function contains(mark: Mark, point: MarkPoint) {
  if (mark.tool === 'bbox' && mark.points.length >= 2) {
    const [a, b] = mark.points
    const x1 = Math.min(a.x, b.x)
    const x2 = Math.max(a.x, b.x)
    const y1 = Math.min(a.y, b.y)
    const y2 = Math.max(a.y, b.y)
    return point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2
  }
  if (mark.tool === 'polygon') return pointInPolygon(point, mark.points)
  return mark.points.some((item, index) => {
    const next = mark.points[index + 1]
    if (!next) return pointNear(point, item, 0.02)
    return distanceToSegment(point, item, next) <= 0.02
  })
}

function pointInPolygon(point: MarkPoint, polygon: MarkPoint[]) {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i]
    const b = polygon[j]
    const cross = a.y > point.y !== b.y > point.y && point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x
    if (cross) inside = !inside
  }
  return inside
}

function distanceToSegment(point: MarkPoint, a: MarkPoint, b: MarkPoint) {
  const dx = b.x - a.x
  const dy = b.y - a.y
  const length = dx * dx + dy * dy
  if (length === 0) return Math.hypot(point.x - a.x, point.y - a.y)
  const t = Math.min(1, Math.max(0, ((point.x - a.x) * dx + (point.y - a.y) * dy) / length))
  return Math.hypot(point.x - (a.x + t * dx), point.y - (a.y + t * dy))
}
