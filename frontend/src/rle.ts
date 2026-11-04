export type RleMask = {
  height: number
  width: number
  counts: string
}

type CachedMask = {
  counts: string
  bits: Uint8Array
  filled: number
}

const bitsCache = new Map<string, CachedMask>()
const sheetCache = new Map<string, HTMLCanvasElement>()

export function decodeCocoCounts(counts: string): number[] {
  const runs: number[] = []
  let cursor = 0
  while (cursor < counts.length) {
    let value = 0
    let group = 0
    let more = 1
    while (more) {
      const chunk = counts.charCodeAt(cursor) - 48
      value += (chunk & 31) * 2 ** (5 * group)
      more = chunk & 32
      cursor += 1
      group += 1
      if (!more && chunk & 16) value -= 2 ** (5 * group)
    }
    if (runs.length > 2) value += runs[runs.length - 2]
    runs.push(value)
  }
  return runs
}

export function decodeRle(counts: string, height: number, width: number): { bits: Uint8Array; filled: number } {
  const bits = new Uint8Array(height * width)
  const runs = decodeCocoCounts(counts)
  let index = 0
  let filled = 0
  let on = 0
  for (const run of runs) {
    const length = run > 0 ? run : 0
    if (on) {
      for (let step = 0; step < length && index < bits.length; step += 1) {
        const row = index % height
        const column = Math.floor(index / height)
        if (column < width) {
          bits[row * width + column] = 1
          filled += 1
        }
        index += 1
      }
    } else {
      index += length
    }
    on = on ? 0 : 1
  }
  return { bits, filled }
}

export function cachedMask(id: string, mask: RleMask): CachedMask {
  const cached = bitsCache.get(id)
  if (cached && cached.counts === mask.counts) return cached
  const decoded = decodeRle(mask.counts, mask.height, mask.width)
  const entry = { counts: mask.counts, ...decoded }
  bitsCache.set(id, entry)
  return entry
}

function encodeCocoCounts(runs: number[]) {
  let text = ''
  for (let index = 0; index < runs.length; index += 1) {
    let value = runs[index]
    if (index > 2) value -= runs[index - 2]
    let more = true
    while (more) {
      let chunk = value & 31
      value >>= 5
      more = chunk & 16 ? value !== -1 : value !== 0
      if (more) chunk |= 32
      text += String.fromCharCode(chunk + 48)
    }
  }
  return text
}

function encodeMaskCounts(bits: Uint8Array, height: number, width: number) {
  const runs: number[] = []
  let on = 0
  let count = 0
  for (let column = 0; column < width; column += 1) {
    for (let row = 0; row < height; row += 1) {
      const bit = bits[row * width + column] ? 1 : 0
      if (bit === on) count += 1
      else {
        runs.push(count)
        on = bit
        count = 1
      }
    }
  }
  runs.push(count)
  return encodeCocoCounts(runs)
}

export function stampMask(
  id: string,
  mask: RleMask,
  shift: { x: number; y: number },
  points: { x: number; y: number }[],
  radiusPx: number,
  frame: { width: number; height: number },
) {
  const cached = cachedMask(id, mask)
  const rx = (radiusPx * mask.width) / Math.max(frame.width, 1)
  const ry = (radiusPx * mask.height) / Math.max(frame.height, 1)
  let changed = false
  for (const center of points) {
    const cx = (center.x - shift.x) * mask.width
    const cy = (center.y - shift.y) * mask.height
    const left = Math.max(0, Math.floor(cx - rx))
    const right = Math.min(mask.width - 1, Math.ceil(cx + rx))
    const top = Math.max(0, Math.floor(cy - ry))
    const bottom = Math.min(mask.height - 1, Math.ceil(cy + ry))
    for (let y = top; y <= bottom; y += 1) {
      for (let x = left; x <= right; x += 1) {
        const dx = (x - cx) / Math.max(rx, 0.001)
        const dy = (y - cy) / Math.max(ry, 0.001)
        if (dx * dx + dy * dy > 1) continue
        const offset = y * mask.width + x
        if (cached.bits[offset]) continue
        cached.bits[offset] = 1
        cached.filled += 1
        changed = true
      }
    }
  }
  if (!changed) return null
  for (const key of sheetCache.keys()) {
    if (key.startsWith(`${id}:`)) sheetCache.delete(key)
  }
  cached.counts = encodeMaskCounts(cached.bits, mask.height, mask.width)
  return cached.counts
}

export function clearMaskCache() {
  bitsCache.clear()
  sheetCache.clear()
}

export function maskContains(
  id: string,
  mask: RleMask,
  shift: { x: number; y: number },
  point: { x: number; y: number },
  frame?: { width: number; height: number },
) {
  const { bits } = cachedMask(id, mask)
  const column = Math.round((point.x - shift.x) * mask.width)
  const row = Math.round((point.y - shift.y) * mask.height)
  const slopX = frame ? Math.ceil((4 * mask.width) / Math.max(frame.width, 1)) : 0
  const slopY = frame ? Math.ceil((4 * mask.height) / Math.max(frame.height, 1)) : 0
  for (let y = row - slopY; y <= row + slopY; y += 1) {
    if (y < 0 || y >= mask.height) continue
    for (let x = column - slopX; x <= column + slopX; x += 1) {
      if (x < 0 || x >= mask.width) continue
      if (bits[y * mask.width + x]) return true
    }
  }
  return false
}

export function punchMask(
  id: string,
  mask: RleMask,
  shift: { x: number; y: number },
  center: { x: number; y: number },
  radiusPx: number,
  frame: { width: number; height: number },
) {
  const cached = cachedMask(id, mask)
  const cx = (center.x - shift.x) * mask.width
  const cy = (center.y - shift.y) * mask.height
  const rx = (radiusPx * mask.width) / Math.max(frame.width, 1)
  const ry = (radiusPx * mask.height) / Math.max(frame.height, 1)
  const left = Math.max(0, Math.floor(cx - rx))
  const right = Math.min(mask.width - 1, Math.ceil(cx + rx))
  const top = Math.max(0, Math.floor(cy - ry))
  const bottom = Math.min(mask.height - 1, Math.ceil(cy + ry))
  let changed = false
  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      const dx = (x - cx) / Math.max(rx, 0.001)
      const dy = (y - cy) / Math.max(ry, 0.001)
      if (dx * dx + dy * dy > 1) continue
      const offset = y * mask.width + x
      if (!cached.bits[offset]) continue
      cached.bits[offset] = 0
      cached.filled -= 1
      changed = true
    }
  }
  if (!changed) return null
  for (const key of sheetCache.keys()) {
    if (key.startsWith(`${id}:`)) sheetCache.delete(key)
  }
  return cached.filled
}

function rgb(color: string) {
  const match = /^#([0-9a-f]{6})$/i.exec(color.trim())
  if (!match) return { r: 255, g: 109, b: 45 }
  const value = Number.parseInt(match[1], 16)
  return { r: (value >> 16) & 255, g: (value >> 8) & 255, b: value & 255 }
}

export function maskSheet(id: string, color: string, rev: number, mask: RleMask) {
  const key = `${id}:${color}:${rev}`
  const cached = sheetCache.get(key)
  if (cached) return cached
  const { bits } = cachedMask(id, mask)
  const canvas = document.createElement('canvas')
  canvas.width = mask.width
  canvas.height = mask.height
  const context = canvas.getContext('2d')
  if (!context) return null
  const image = context.createImageData(mask.width, mask.height)
  const fill = rgb(color)
  for (let index = 0; index < bits.length; index += 1) {
    if (!bits[index]) continue
    const offset = index * 4
    image.data[offset] = fill.r
    image.data[offset + 1] = fill.g
    image.data[offset + 2] = fill.b
    image.data[offset + 3] = 255
  }
  context.putImageData(image, 0, 0)
  sheetCache.set(key, canvas)
  return canvas
}
