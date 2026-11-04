import type { LabelTool } from './projects'
import { maskContains, punchMask, stampMask, type RleMask } from './rle'

export type MarkPoint = { x: number; y: number }

export type MarkMask = RleMask & { box: Box }

export type Mark = {
  id: string
  tool: LabelTool
  name: string
  color: string
  points: MarkPoint[]
  size?: number
  cuts?: number[]
  mask?: MarkMask
  maskRev?: number
}

export type Box = { x: number; y: number; w: number; h: number }

export const BOX_HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'] as const
export type HandleName = (typeof BOX_HANDLES)[number]

export type FrameSize = { width: number; height: number }

function storageKey(projectId: string) {
  return `labeling.marks.${projectId}`
}

function readBox(value: unknown): Box | null {
  if (!value || typeof value !== 'object') return null
  const box = value as Partial<Box>
  if (typeof box.x !== 'number' || typeof box.y !== 'number' || typeof box.w !== 'number' || typeof box.h !== 'number') return null
  return { x: box.x, y: box.y, w: box.w, h: box.h }
}

function readMask(value: unknown): MarkMask | undefined {
  if (!value || typeof value !== 'object') return undefined
  const row = value as Partial<MarkMask>
  const box = readBox(row.box)
  if (!box || typeof row.counts !== 'string' || typeof row.height !== 'number' || typeof row.width !== 'number') return undefined
  if (row.height <= 0 || row.width <= 0 || !row.counts) return undefined
  return { height: row.height, width: row.width, counts: row.counts, box }
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
  const cuts = Array.isArray(row.cuts)
    ? row.cuts.filter((cut): cut is number => typeof cut === 'number' && cut > 0 && cut < points.length)
    : undefined
  const mask = readMask(row.mask)
  return {
    id: typeof row.id === 'string' ? row.id : crypto.randomUUID(),
    tool: row.tool,
    name: row.name,
    color: row.color,
    points,
    size,
    ...(cuts && cuts.length > 0 ? { cuts } : {}),
    ...(mask ? { mask } : {}),
  }
}

export function brushStrokes(points: MarkPoint[], cuts?: number[]): MarkPoint[][] {
  if (points.length === 0) return []
  const starts = [...new Set(cuts ?? [])].filter((cut) => cut > 0 && cut < points.length).sort((a, b) => a - b)
  const strokes: MarkPoint[][] = []
  let start = 0
  for (const cut of starts) {
    strokes.push(points.slice(start, cut))
    start = cut
  }
  strokes.push(points.slice(start))
  return strokes.filter((stroke) => stroke.length > 0)
}

function readFileMark(value: unknown): Mark | null {
  if (!value || typeof value !== 'object') return null
  const row = value as Record<string, unknown>
  const color = typeof row.color === 'string' ? row.color : '#ff6d2d'
  return readMark({ ...row, color })
}

function addMark(grouped: Record<string, Mark[]>, image: string, mark: Mark | null) {
  const key = image.trim().replace(/\\/g, '/')
  if (!key || !mark) return
  const list = grouped[key] ?? []
  list.push(mark)
  grouped[key] = list
}

export function marksByImage(data: unknown): Record<string, Mark[]> {
  const grouped: Record<string, Mark[]> = {}
  if (!data || typeof data !== 'object') return grouped
  const root = data as Record<string, unknown>
  const source = 'annotations' in root ? root.annotations : root

  if (Array.isArray(source)) {
    for (const item of source) {
      if (!item || typeof item !== 'object') continue
      const row = item as Record<string, unknown>
      const image = typeof row.image === 'string' ? row.image : typeof row.file === 'string' ? row.file : ''
      addMark(grouped, image, readFileMark(item))
    }
    return grouped
  }

  if (source && typeof source === 'object') {
    for (const [image, value] of Object.entries(source)) {
      if (!Array.isArray(value)) continue
      for (const item of value) addMark(grouped, image, readFileMark(item))
    }
  }
  return grouped
}

const MASK_COLORS = ['#ff6d2d', '#3b82f6', '#22c55e', '#eab308', '#a855f7', '#ef4444', '#14b8a6', '#ec4899']

export function marksFromCoco(data: unknown): Record<string, Mark[]> {
  const grouped: Record<string, Mark[]> = {}
  if (!data || typeof data !== 'object') return grouped
  const root = data as {
    images?: unknown
    annotations?: unknown
    categories?: unknown
  }
  if (!Array.isArray(root.images) || !Array.isArray(root.annotations)) return grouped

  const images = new Map<number, { file: string; width: number; height: number }>()
  for (const item of root.images) {
    if (!item || typeof item !== 'object') continue
    const row = item as { id?: unknown; file_name?: unknown; width?: unknown; height?: unknown }
    if (typeof row.id !== 'number' || typeof row.file_name !== 'string') continue
    if (typeof row.width !== 'number' || typeof row.height !== 'number' || row.width <= 0 || row.height <= 0) continue
    images.set(row.id, { file: row.file_name.replace(/\\/g, '/'), width: row.width, height: row.height })
  }

  const names = new Map<number, string>()
  if (Array.isArray(root.categories)) {
    for (const item of root.categories) {
      if (!item || typeof item !== 'object') continue
      const row = item as { id?: unknown; name?: unknown }
      if (typeof row.id !== 'number' || typeof row.name !== 'string' || !row.name) continue
      names.set(row.id, row.name)
    }
  }
  const colors = new Map<number, string>()
  let colorIndex = 0
  for (const id of names.keys()) {
    colors.set(id, MASK_COLORS[colorIndex % MASK_COLORS.length])
    colorIndex += 1
  }

  for (const item of root.annotations) {
    if (!item || typeof item !== 'object') continue
    const row = item as {
      id?: unknown
      image_id?: unknown
      category_id?: unknown
      bbox?: unknown
      segmentation?: unknown
    }
    if (typeof row.image_id !== 'number') continue
    const image = images.get(row.image_id)
    const segmentation = row.segmentation
    if (!image || !segmentation || typeof segmentation !== 'object' || Array.isArray(segmentation)) continue
    const rle = segmentation as { size?: unknown; counts?: unknown }
    if (typeof rle.counts !== 'string' || !Array.isArray(rle.size) || rle.size.length < 2) continue
    const height = rle.size[0]
    const width = rle.size[1]
    if (typeof height !== 'number' || typeof width !== 'number' || height <= 0 || width <= 0) continue
    if (!Array.isArray(row.bbox) || row.bbox.length < 4) continue
    const [x, y, w, h] = row.bbox
    if (typeof x !== 'number' || typeof y !== 'number' || typeof w !== 'number' || typeof h !== 'number') continue
    const category = typeof row.category_id === 'number' ? row.category_id : -1
    const box = {
      x: x / image.width,
      y: y / image.height,
      w: w / image.width,
      h: h / image.height,
    }
    addMark(grouped, image.file, {
      id: typeof row.id === 'number' ? `coco-${row.id}` : crypto.randomUUID(),
      tool: 'brush',
      name: names.get(category) ?? 'object',
      color: colors.get(category) ?? MASK_COLORS[0],
      points: [
        { x: box.x, y: box.y },
        { x: box.x + box.w, y: box.y + box.h },
      ],
      mask: { height, width, counts: rle.counts, box },
    })
  }
  return grouped
}

export function marksForImage(all: Record<string, Mark[]>, imageName: string): Mark[] {
  const exact = all[imageName]
  if (exact) return exact
  const base = imageName.slice(imageName.lastIndexOf('/') + 1)
  if (base !== imageName && all[base]) return all[base]
  return []
}

export function loadStoredMarks(projectId: string): Record<string, Mark[]> {
  try {
    const raw = localStorage.getItem(storageKey(projectId))
    if (!raw) return {}
    const all = JSON.parse(raw) as Record<string, unknown>
    const grouped: Record<string, Mark[]> = {}
    for (const [name, value] of Object.entries(all)) {
      if (!Array.isArray(value)) continue
      const marks = value.flatMap((item) => {
        const mark = readMark(item)
        return mark ? [mark] : []
      })
      if (marks.length > 0) grouped[name] = marks
    }
    return grouped
  } catch {
    return {}
  }
}

export function loadMarks(projectId: string, imageName: string): Mark[] {
  return loadStoredMarks(projectId)[imageName] ?? []
}

export function labeledImages(projectId: string): Set<string> {
  try {
    const raw = localStorage.getItem(storageKey(projectId))
    if (!raw) return new Set()
    const all = JSON.parse(raw) as Record<string, unknown>
    return new Set(
      Object.keys(all).filter((name) => {
        const list = all[name]
        return Array.isArray(list) && list.length > 0
      }),
    )
  } catch {
    return new Set()
  }
}

export function retargetStoredMarks(projectId: string, tool: LabelTool) {
  try {
    const raw = localStorage.getItem(storageKey(projectId))
    if (!raw) return
    const all = JSON.parse(raw) as Record<string, unknown>
    for (const name of Object.keys(all)) {
      if (!Array.isArray(all[name])) continue
      all[name] = (all[name] as unknown[]).flatMap((item) => {
        const mark = readMark(item)
        if (!mark) return []
        const converted = retargetMark(mark, tool)
        return converted ? [converted] : []
      })
    }
    localStorage.setItem(storageKey(projectId), JSON.stringify(all))
  } catch {
    return
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

function boundsOf(points: MarkPoint[]): Box | null {
  if (points.length === 0) return null
  let minX = 1
  let minY = 1
  let maxX = 0
  let maxY = 0
  for (const point of points) {
    minX = Math.min(minX, point.x)
    minY = Math.min(minY, point.y)
    maxX = Math.max(maxX, point.x)
    maxY = Math.max(maxY, point.y)
  }
  if (maxX - minX <= 0 || maxY - minY <= 0) return null
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY }
}

function cornersOf(box: Box): MarkPoint[] {
  return [
    { x: box.x, y: box.y },
    { x: box.x + box.w, y: box.y },
    { x: box.x + box.w, y: box.y + box.h },
    { x: box.x, y: box.y + box.h },
  ]
}

export function retargetMark(mark: Mark, tool: LabelTool): Mark | null {
  if (mark.tool === tool) return mark
  const box = mark.tool === 'bbox' ? boxOf(mark.points) : boundsOf(mark.points)
  if (tool === 'bbox') {
    if (!box) return null
    return { id: mark.id, tool, name: mark.name, color: mark.color, points: boxPoints(box) }
  }
  const points = mark.tool === 'bbox' && box ? cornersOf(box) : mark.points
  if (tool === 'polygon') {
    if (points.length < 3) return null
    return { id: mark.id, tool, name: mark.name, color: mark.color, points }
  }
  if (points.length === 0) return null
  return { id: mark.id, tool, name: mark.name, color: mark.color, points, size: mark.size ?? 18 }
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

export function hitPolygonEdge(points: MarkPoint[], point: MarkPoint, frame: FrameSize, slop = 8) {
  if (points.length < 2) return null
  let best: { index: number; point: MarkPoint; distance: number } | null = null
  for (let index = 0; index < points.length; index += 1) {
    const next = points[(index + 1) % points.length]
    const hit = closestOnSegment(point, points[index], next, frame)
    if (hit.distance > slop) continue
    if (pixelDistance(hit.point, points[index], frame) <= 8 || pixelDistance(hit.point, next, frame) <= 8) continue
    if (best && hit.distance >= best.distance) continue
    best = { index, point: hit.point, distance: hit.distance }
  }
  return best ? { index: best.index, point: best.point } : null
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
    if (mark.mask) {
      const filled = punchMask(mark.id, mark.mask, maskOffset(mark), center, radiusPx, frame)
      if (filled === null) next.push(mark)
      else if (filled > 0) next.push({ ...mark, maskRev: (mark.maskRev ?? 0) + 1 })
      continue
    }
    const reach = radiusPx + (mark.size ?? 18) / 2
    const kept: MarkPoint[][] = []
    for (const stroke of brushStrokes(mark.points, mark.cuts)) {
      const groups: MarkPoint[][] = [[]]
      for (const point of stroke) {
        if (pixelDistance(point, center, frame) <= reach) {
          if (groups[groups.length - 1].length > 0) groups.push([])
        } else {
          groups[groups.length - 1].push(point)
        }
      }
      for (const group of groups) {
        if (group.length === 0) continue
        kept.push(group.length === 1 ? [group[0], { ...group[0] }] : group)
      }
    }
    if (kept.length === 0) continue
    let cursor = 0
    const cuts: number[] = []
    for (const group of kept) {
      if (cursor > 0) cuts.push(cursor)
      cursor += group.length
    }
    next.push({
      ...mark,
      points: kept.flat(),
      ...(cuts.length > 0 ? { cuts } : { cuts: undefined }),
    })
  }
  return next
}

function growBox(box: Box, points: MarkPoint[], padX: number, padY: number): Box {
  let x1 = box.x
  let y1 = box.y
  let x2 = box.x + box.w
  let y2 = box.y + box.h
  for (const point of points) {
    x1 = Math.min(x1, point.x - padX)
    y1 = Math.min(y1, point.y - padY)
    x2 = Math.max(x2, point.x + padX)
    y2 = Math.max(y2, point.y + padY)
  }
  x1 = Math.max(0, x1)
  y1 = Math.max(0, y1)
  x2 = Math.min(1, x2)
  y2 = Math.min(1, y2)
  return { x: x1, y: y1, w: Math.max(0, x2 - x1), h: Math.max(0, y2 - y1) }
}

export function addBrushStroke(mark: Mark, points: MarkPoint[], radiusPx: number, frame: FrameSize): Mark | null {
  if (mark.tool !== 'brush' || points.length === 0) return null
  if (mark.mask) {
    const shift = maskOffset(mark)
    const counts = stampMask(mark.id, mark.mask, shift, points, radiusPx, frame)
    if (!counts) return null
    const padX = radiusPx / Math.max(frame.width, 1)
    const padY = radiusPx / Math.max(frame.height, 1)
    const current = boxOf(mark.points) ?? mark.mask.box
    const nextBox = growBox(current, points, padX, padY)
    const nextOrigin = growBox(
      mark.mask.box,
      points.map((point) => ({ x: point.x - shift.x, y: point.y - shift.y })),
      padX,
      padY,
    )
    return {
      ...mark,
      points: boxPoints(nextBox),
      mask: { ...mark.mask, counts, box: nextOrigin },
      maskRev: (mark.maskRev ?? 0) + 1,
    }
  }
  const cut = mark.points.length
  return {
    ...mark,
    points: [...mark.points, ...points],
    cuts: [...(mark.cuts ?? []), cut],
  }
}

export function maskOffset(mark: Mark) {
  const origin = mark.mask?.box
  const current = boxOf(mark.points)
  if (!origin || !current) return { x: 0, y: 0 }
  return { x: current.x - origin.x, y: current.y - origin.y }
}

function hits(mark: Mark, point: MarkPoint, frame?: FrameSize) {
  if (mark.mask) return maskContains(mark.id, mark.mask, maskOffset(mark), point, frame)
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
  return brushStrokes(mark.points, mark.cuts).some((stroke) => {
    if (!frame) return stroke.some((item) => Math.hypot(item.x - point.x, item.y - point.y) <= reach)
    return edgeNear(stroke, point, frame, reach, false)
  })
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

function closestOnSegment(point: MarkPoint, a: MarkPoint, b: MarkPoint, frame: FrameSize) {
  const ax = a.x * frame.width
  const ay = a.y * frame.height
  const bx = b.x * frame.width
  const by = b.y * frame.height
  const px = point.x * frame.width
  const py = point.y * frame.height
  const dx = bx - ax
  const dy = by - ay
  const length = dx * dx + dy * dy
  const t = length === 0 ? 0 : Math.min(1, Math.max(0, ((px - ax) * dx + (py - ay) * dy) / length))
  const x = ax + t * dx
  const y = ay + t * dy
  return {
    t,
    distance: Math.hypot(px - x, py - y),
    point: {
      x: frame.width ? x / frame.width : a.x,
      y: frame.height ? y / frame.height : a.y,
    },
  }
}

function segmentDistance(point: MarkPoint, a: MarkPoint, b: MarkPoint, frame: FrameSize) {
  return closestOnSegment(point, a, b, frame).distance
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
