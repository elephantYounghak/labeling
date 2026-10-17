import type { LabelTool } from './projects'

export type MarkPoint = { x: number; y: number }

export type Mark = {
  id: string
  tool: LabelTool
  name: string
  color: string
  points: MarkPoint[]
  size?: number
}

export type Box = { x: number; y: number; w: number; h: number }

export const BOX_HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'] as const
export type HandleName = (typeof BOX_HANDLES)[number]

export type FrameSize = { width: number; height: number }

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
  const size = typeof row.size === 'number' && row.size > 0 ? row.size : undefined
  return {
    id: typeof row.id === 'string' ? row.id : crypto.randomUUID(),
    tool: row.tool,
    name: row.name,
    color: row.color,
    points,
    size,
  }
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

export function clamp01(value: number) {
  return Math.min(1, Math.max(0, value))
}

export function clampPoint(point: MarkPoint): MarkPoint {
  return { x: clamp01(point.x), y: clamp01(point.y) }
}

export function pixelDistance(a: MarkPoint, b: MarkPoint, frame: FrameSize) {
  return Math.hypot((a.x - b.x) * frame.width, (a.y - b.y) * frame.height)
}

export function boxOf(points: MarkPoint[]): Box | null {
  if (points.length < 2) return null
  const [a, b] = points
  const x = Math.min(a.x, b.x)
  const y = Math.min(a.y, b.y)
  return { x, y, w: Math.abs(b.x - a.x), h: Math.abs(b.y - a.y) }
}

export function boxPoints(box: Box): MarkPoint[] {
  return [
    { x: box.x, y: box.y },
    { x: box.x + box.w, y: box.y + box.h },
  ]
}

export function handlePoint(box: Box, handle: HandleName): MarkPoint {
  const x = handle.includes('w') ? box.x : handle.includes('e') ? box.x + box.w : box.x + box.w / 2
  const y = handle.includes('n') ? box.y : handle.includes('s') ? box.y + box.h : box.y + box.h / 2
  return { x, y }
}

export function hitBoxHandle(box: Box, point: MarkPoint, frame: FrameSize, slop = 8): HandleName | null {
  const order: HandleName[] = ['nw', 'ne', 'se', 'sw', 'n', 'e', 's', 'w']
  for (const handle of order) {
    if (pixelDistance(handlePoint(box, handle), point, frame) <= slop) return handle
  }
  return null
}

export function hitVertex(points: MarkPoint[], point: MarkPoint, frame: FrameSize, slop = 8) {
  for (let index = 0; index < points.length; index += 1) {
    if (pixelDistance(points[index], point, frame) <= slop) return index
  }
  return -1
}

export function squarePoint(start: MarkPoint, point: MarkPoint, frame: FrameSize): MarkPoint {
  const dx = (point.x - start.x) * frame.width
  const dy = (point.y - start.y) * frame.height
  const sx = dx < 0 ? -1 : 1
  const sy = dy < 0 ? -1 : 1
  const maxX = (sx < 0 ? start.x : 1 - start.x) * frame.width
  const maxY = (sy < 0 ? start.y : 1 - start.y) * frame.height
  const side = Math.min(Math.max(Math.abs(dx), Math.abs(dy)), Math.max(0, maxX), Math.max(0, maxY))
  return {
    x: start.x + (sx * side) / Math.max(frame.width, 1),
    y: start.y + (sy * side) / Math.max(frame.height, 1),
  }
}

export function translateInside(points: MarkPoint[], dx: number, dy: number): MarkPoint[] {
  if (points.length === 0) return points
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const point of points) {
    minX = Math.min(minX, point.x)
    minY = Math.min(minY, point.y)
    maxX = Math.max(maxX, point.x)
    maxY = Math.max(maxY, point.y)
  }
  const cdx = Math.min(Math.max(dx, -minX), 1 - maxX)
  const cdy = Math.min(Math.max(dy, -minY), 1 - maxY)
  return points.map((point) => ({ x: point.x + cdx, y: point.y + cdy }))
}

export function resizeBox(box: Box, handle: HandleName, point: MarkPoint, frame: FrameSize, keepRatio: boolean): Box {
  let x1 = box.x
  let y1 = box.y
  let x2 = box.x + box.w
  let y2 = box.y + box.h
  const ratio = keepRatio && box.w > 0 && box.h > 0 && handle.length === 2 ? (box.w * frame.width) / (box.h * frame.height) : 0

  if (ratio) {
    const ax = handle.includes('w') ? x2 : x1
    const ay = handle.includes('n') ? y2 : y1
    const signX = handle.includes('w') ? -1 : 1
    const signY = handle.includes('n') ? -1 : 1
    let width = Math.abs(point.x - ax) * frame.width
    let height = Math.abs(point.y - ay) * frame.height
    if (height === 0 || width / height > ratio) height = width / ratio
    else width = height * ratio
    const maxW = (signX < 0 ? ax : 1 - ax) * frame.width
    const maxH = (signY < 0 ? ay : 1 - ay) * frame.height
    const limit = Math.min(width > 0 ? maxW / width : 1, height > 0 ? maxH / height : 1, 1)
    width *= Math.max(0, limit)
    height *= Math.max(0, limit)
    const nextX = ax + (signX * width) / Math.max(frame.width, 1)
    const nextY = ay + (signY * height) / Math.max(frame.height, 1)
    x1 = Math.min(ax, nextX)
    x2 = Math.max(ax, nextX)
    y1 = Math.min(ay, nextY)
    y2 = Math.max(ay, nextY)
  } else {
    if (handle.includes('w')) x1 = point.x
    if (handle.includes('e')) x2 = point.x
    if (handle.includes('n')) y1 = point.y
    if (handle.includes('s')) y2 = point.y
  }

  const left = clamp01(Math.min(x1, x2))
  const right = clamp01(Math.max(x1, x2))
  const top = clamp01(Math.min(y1, y2))
  const bottom = clamp01(Math.max(y1, y2))
  return { x: left, y: top, w: right - left, h: bottom - top }
}

export function labelAnchor(mark: Mark): MarkPoint | null {
  if (mark.tool === 'bbox') {
    const box = boxOf(mark.points)
    return box ? { x: box.x, y: box.y } : null
  }
  if (mark.points.length === 0) return null
  return mark.points.reduce((top, point) => (point.y < top.y ? point : top))
}

export function hitMark(marks: Mark[], point: MarkPoint, frame?: FrameSize): Mark | null {
  for (let index = marks.length - 1; index >= 0; index -= 1) {
    if (hits(marks[index], point, frame)) return marks[index]
  }
  return null
}

export function eraseBrush(marks: Mark[], center: MarkPoint, radiusPx: number, frame: FrameSize): Mark[] {
  const next: Mark[] = []
  for (const mark of marks) {
    if (mark.tool !== 'brush') {
      next.push(mark)
      continue
    }
    const reach = radiusPx + (mark.size ?? 18) / 2
    const groups: MarkPoint[][] = [[]]
    for (const point of mark.points) {
      if (pixelDistance(point, center, frame) <= reach) {
        if (groups[groups.length - 1].length > 0) groups.push([])
      } else {
        groups[groups.length - 1].push(point)
      }
    }
    let usedId = false
    for (const group of groups) {
      if (group.length === 0) continue
      next.push({
        ...mark,
        id: usedId ? crypto.randomUUID() : mark.id,
        points: group.length === 1 ? [group[0], { ...group[0] }] : group,
      })
      usedId = true
    }
  }
  return next
}

function hits(mark: Mark, point: MarkPoint, frame?: FrameSize) {
  if (mark.tool === 'bbox') {
    const box = boxOf(mark.points)
    if (!box) return false
    const padX = frame ? 4 / frame.width : 0
    const padY = frame ? 4 / frame.height : 0
    return point.x >= box.x - padX && point.x <= box.x + box.w + padX && point.y >= box.y - padY && point.y <= box.y + box.h + padY
  }
  if (mark.tool === 'polygon') {
    if (pointInPolygon(point, mark.points)) return true
    if (!frame) return false
    return edgeNear(mark.points, point, frame, 6, true)
  }
  const reach = frame ? (mark.size ?? 18) / 2 + 4 : 0.02
  if (!frame) return mark.points.some((item) => Math.hypot(item.x - point.x, item.y - point.y) <= reach)
  return edgeNear(mark.points, point, frame, reach, false)
}

function edgeNear(points: MarkPoint[], point: MarkPoint, frame: FrameSize, slop: number, closed: boolean) {
  if (points.length === 0) return false
  if (points.length === 1) return pixelDistance(points[0], point, frame) <= slop
  const last = closed ? points.length : points.length - 1
  for (let index = 0; index < last; index += 1) {
    const next = points[(index + 1) % points.length]
    if (segmentDistance(point, points[index], next, frame) <= slop) return true
  }
  return false
}

function segmentDistance(point: MarkPoint, a: MarkPoint, b: MarkPoint, frame: FrameSize) {
  const ax = a.x * frame.width
  const ay = a.y * frame.height
  const bx = b.x * frame.width
  const by = b.y * frame.height
  const px = point.x * frame.width
  const py = point.y * frame.height
  const dx = bx - ax
  const dy = by - ay
  const length = dx * dx + dy * dy
  if (length === 0) return Math.hypot(px - ax, py - ay)
  const t = Math.min(1, Math.max(0, ((px - ax) * dx + (py - ay) * dy) / length))
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
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
