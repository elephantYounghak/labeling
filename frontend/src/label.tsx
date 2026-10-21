import { useEffect, useRef, useState, type MouseEvent, type PointerEvent as ReactPointerEvent } from 'react'
import {
  BOX_HANDLES,
  boxOf,
  boxPoints,
  clamp01,
  clampPoint,
  eraseBrush,
  handlePoint,
  hitBoxHandle,
  hitMark,
  hitVertex,
  labelAnchor,
  loadMarks,
  pixelDistance,
  resizeBox,
  saveMarks,
  squarePoint,
  translateInside,
  type Box,
  type FrameSize,
  type HandleName,
  type Mark,
  type MarkPoint,
} from './marks'
import { detectApi, type DetectMode } from './detect'
import type { LabelClass, LabelTool } from './projects'

type Frame = FrameSize & { left: number; top: number }

type Gesture =
  | { kind: 'bbox'; start: MarkPoint }
  | { kind: 'brush'; points: MarkPoint[]; moved: boolean }
  | { kind: 'erase'; last: MarkPoint }
  | { kind: 'move'; id: string; origin: MarkPoint; base: MarkPoint[] }
  | { kind: 'resize'; id: string; handle: HandleName; box: Box }
  | { kind: 'vertex'; id: string; index: number }

const TOOLS: { id: LabelTool; label: string; key: string }[] = [
  { id: 'bbox', label: 'bbox', key: 'B' },
  { id: 'polygon', label: 'polygon', key: 'P' },
  { id: 'brush', label: 'brush', key: 'M' },
]

const RESIZE_CURSOR: Record<HandleName, string> = {
  nw: 'nwse-resize',
  se: 'nwse-resize',
  ne: 'nesw-resize',
  sw: 'nesw-resize',
  n: 'ns-resize',
  s: 'ns-resize',
  e: 'ew-resize',
  w: 'ew-resize',
}

let rememberedBrush = 18
let rememberedErase = false

type Camera = { zoom: number; x: number; y: number }

function clampPan(x: number, y: number, zoom: number, frame: Frame, viewWidth: number, viewHeight: number) {
  const width = frame.width * zoom
  const height = frame.height * zoom
  let left = frame.left + x
  let top = frame.top + y
  if (width >= viewWidth) left = Math.min(0, Math.max(viewWidth - width, left))
  else left = Math.min(viewWidth - width, Math.max(0, left))
  if (height >= viewHeight) top = Math.min(0, Math.max(viewHeight - height, top))
  else top = Math.min(viewHeight - height, Math.max(0, top))
  return { x: left - frame.left, y: top - frame.top }
}

export function LabelStage({
  projectId,
  imageName,
  imageUrl,
  imageFile,
  tool,
  opacity,
  labelClass,
  classes,
  onClassIndex,
  onTool,
  modelId,
  modelName,
  folderPath,
  detectMode,
  onDetectMode,
}: {
  projectId: string
  imageName: string
  imageUrl: string
  imageFile: File
  tool: LabelTool
  opacity: number
  labelClass: LabelClass | null
  classes: LabelClass[]
  onClassIndex: (index: number) => void
  onTool: (tool: LabelTool) => void
  modelId: string
  modelName: string
  folderPath: string
  detectMode: DetectMode
  onDetectMode: (mode: DetectMode) => void
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const brushCanvasRef = useRef<HTMLCanvasElement>(null)
  const [frame, setFrame] = useState<Frame>({ left: 0, top: 0, width: 0, height: 0 })
  const [natural, setNatural] = useState({ width: 0, height: 0 })
  const [marks, setMarks] = useState<Mark[]>(() => loadMarks(projectId, imageName))
  const [draft, setDraft] = useState<MarkPoint[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [pointer, setPointer] = useState<MarkPoint | null>(null)
  const [gesture, setGesture] = useState<Gesture | null>(null)
  const [brushSize, setBrushSize] = useState(rememberedBrush)
  const [erasing, setErasing] = useState(rememberedErase)
  const [camera, setCamera] = useState<Camera>({ zoom: 1, x: 0, y: 0 })
  const [awaitingClick, setAwaitingClick] = useState(false)
  const [detectBusy, setDetectBusy] = useState(false)
  const [detectNote, setDetectNote] = useState('')
  const [detectFailed, setDetectFailed] = useState(false)
  const [labelOpen, setLabelOpen] = useState(false)
  const [seenImage, setSeenImage] = useState(imageUrl)
  if (seenImage !== imageUrl) {
    setSeenImage(imageUrl)
    setCamera({ zoom: 1, x: 0, y: 0 })
  }

  const marksRef = useRef(marks)
  const draftRef = useRef(draft)
  const selectedRef = useRef(selectedId)
  const gestureRef = useRef(gesture)
  const beforeRef = useRef<Mark[] | null>(null)
  const pastRef = useRef<Mark[][]>([])
  const futureRef = useRef<Mark[][]>([])
  const frameRef = useRef(frame)
  const toolRef = useRef(tool)
  const labelRef = useRef(labelClass)
  const classesRef = useRef(classes)
  const onClassRef = useRef(onClassIndex)
  const onToolRef = useRef(onTool)
  const brushRef = useRef(brushSize)
  const eraseRef = useRef(erasing)
  const opacityRef = useRef(opacity)
  const cameraRef = useRef(camera)
  const finishRef = useRef<(points: MarkPoint[]) => void>(() => {})
  const awaitingRef = useRef(false)
  const detectModeRef = useRef(detectMode)
  const pointerRef = useRef<MarkPoint | null>(null)
  const labelMenuRef = useRef<HTMLDivElement>(null)
  const labelOpenRef = useRef(false)
  const detectBusyRef = useRef(false)
  const closedAt = useRef(0)
  const toolPrev = useRef(tool)

  function replaceDraft(points: MarkPoint[]) {
    draftRef.current = points
    setDraft(points)
  }

  function setMarksNow(next: Mark[]) {
    marksRef.current = next
    setMarks(next)
  }

  function select(id: string | null) {
    selectedRef.current = id
    setSelectedId(id)
  }

  function setGestureNow(next: Gesture | null) {
    gestureRef.current = next
    setGesture(next)
  }

  function pushPast(before: Mark[]) {
    pastRef.current.push(before)
    if (pastRef.current.length > 100) pastRef.current.shift()
    futureRef.current = []
  }

  function beginHistory() {
    if (!beforeRef.current) beforeRef.current = marksRef.current
  }

  function commitHistory() {
    const before = beforeRef.current
    beforeRef.current = null
    if (!before || JSON.stringify(before) === JSON.stringify(marksRef.current)) return
    pushPast(before)
  }

  function finishMark(nextTool: LabelTool, points: MarkPoint[], size?: number) {
    const label = labelRef.current
    if (!label?.name || points.length === 0) return
    const before = marksRef.current
    const mark: Mark = {
      id: crypto.randomUUID(),
      tool: nextTool,
      name: label.name,
      color: label.color,
      points,
      ...(size ? { size } : {}),
    }
    setMarksNow([...before, mark])
    pushPast(before)
    select(mark.id)
  }

  function finishPolygon(points: MarkPoint[]) {
    if (points.length < 3) return
    closedAt.current = performance.now()
    replaceDraft([])
    finishMark('polygon', points.map((point) => clampPoint(point)))
  }

  useEffect(() => {
    marksRef.current = marks
    draftRef.current = draft
    selectedRef.current = selectedId
    frameRef.current = frame
    toolRef.current = tool
    labelRef.current = labelClass
    classesRef.current = classes
    onClassRef.current = onClassIndex
    onToolRef.current = onTool
    brushRef.current = brushSize
    eraseRef.current = erasing
    opacityRef.current = opacity
    cameraRef.current = camera
    finishRef.current = finishPolygon
    awaitingRef.current = awaitingClick
    detectModeRef.current = detectMode
    pointerRef.current = pointer
    detectBusyRef.current = detectBusy
    paintBrushLayer()
  })

  function removeMark(id: string) {
    const before = marksRef.current
    const next = before.filter((mark) => mark.id !== id)
    if (next.length === before.length) return
    setMarksNow(next)
    pushPast(before)
    if (selectedRef.current === id) select(null)
  }

  function undo() {
    if (gestureRef.current) return
    if (draftRef.current.length > 0) {
      replaceDraft(draftRef.current.slice(0, -1))
      return
    }
    const prev = pastRef.current.pop()
    if (!prev) return
    futureRef.current.push(marksRef.current)
    setMarksNow(prev)
    select(null)
  }

  function redo() {
    if (gestureRef.current) return
    const next = futureRef.current.pop()
    if (!next) return
    pastRef.current.push(marksRef.current)
    setMarksNow(next)
    select(null)
  }

  function changeBrush(next: number) {
    const size = Math.min(72, Math.max(4, next))
    rememberedBrush = size
    brushRef.current = size
    setBrushSize(size)
  }

  function toggleErase() {
    setErasing((value) => {
      rememberedErase = !value
      eraseRef.current = !value
      return !value
    })
  }

  useEffect(() => {
    saveMarks(projectId, imageName, marks)
  }, [projectId, imageName, marks])

  useEffect(() => {
    if (detectMode !== 'click') return
    replaceDraft([])
    setGestureNow(null)
    beforeRef.current = null
  }, [detectMode])

  useEffect(() => {
    if (toolPrev.current === tool) return
    toolPrev.current = tool
    if (beforeRef.current) {
      marksRef.current = beforeRef.current
      setMarks(beforeRef.current)
      beforeRef.current = null
    }
    draftRef.current = []
    setDraft([])
    gestureRef.current = null
    setGesture(null)
  }, [tool])

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return
      const key = event.key.toLowerCase()
      if (event.key === 'Escape') {
        if (labelOpenRef.current) {
          labelOpenRef.current = false
          setLabelOpen(false)
          return
        }
        if (awaitingRef.current) {
          awaitingRef.current = false
          setAwaitingClick(false)
          setDetectNote('')
          setDetectFailed(false)
        }
        replaceDraft([])
        if (beforeRef.current) setMarksNow(beforeRef.current)
        beforeRef.current = null
        setGestureNow(null)
        select(null)
        return
      }
      if (!event.ctrlKey && !event.metaKey && !event.altKey && /^[1-9]$/.test(event.key)) {
        const index = Number(event.key) - 1
        if (index < classesRef.current.length) onClassRef.current(index)
        return
      }
      if (detectModeRef.current === 'click') return
      if (event.key === 'Enter' && toolRef.current === 'polygon' && draftRef.current.length >= 3) {
        event.preventDefault()
        finishRef.current(draftRef.current)
        return
      }
      if ((event.ctrlKey || event.metaKey) && key === 'z') {
        event.preventDefault()
        if (event.shiftKey) redo()
        else undo()
        return
      }
      if ((event.ctrlKey || event.metaKey) && key === 'y') {
        event.preventDefault()
        redo()
        return
      }
      if (event.key === 'Backspace' && toolRef.current === 'polygon' && draftRef.current.length > 0) {
        event.preventDefault()
        replaceDraft(draftRef.current.slice(0, -1))
        return
      }
      if ((event.key === 'Delete' || event.key === 'Backspace') && selectedRef.current) {
        event.preventDefault()
        removeMark(selectedRef.current)
        return
      }
      if (!event.ctrlKey && !event.metaKey && !event.altKey && (key === 'b' || key === 'p' || key === 'm')) {
        onToolRef.current(key === 'b' ? 'bbox' : key === 'p' ? 'polygon' : 'brush')
        return
      }
      if (toolRef.current === 'brush' && !event.ctrlKey && !event.metaKey && key === 'e') {
        toggleErase()
        return
      }
      if (toolRef.current === 'brush' && (event.key === '[' || event.key === ']')) {
        changeBrush(brushRef.current + (event.key === ']' ? 2 : -2))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    if (!labelOpen) return
    function onPointer(event: PointerEvent) {
      if (labelMenuRef.current?.contains(event.target as Node)) return
      labelOpenRef.current = false
      setLabelOpen(false)
    }
    window.addEventListener('pointerdown', onPointer)
    return () => window.removeEventListener('pointerdown', onPointer)
  }, [labelOpen])

  useEffect(() => {
    const node = stageRef.current
    if (!node) return
    function onWheel(event: WheelEvent) {
      const stage = stageRef.current
      const box = frameRef.current
      if (!stage || box.width <= 0 || box.height <= 0) return
      const stageRect = stage.getBoundingClientRect()
      const current = cameraRef.current
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault()
        const cursorX = event.clientX - stageRect.left
        const cursorY = event.clientY - stageRect.top
        const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12
        let nextZoom = Math.min(8, Math.max(1, current.zoom * factor))
        if (nextZoom < 1.02) nextZoom = 1
        if (nextZoom === current.zoom) return
        const worldX = (cursorX - box.left - current.x) / current.zoom
        const worldY = (cursorY - box.top - current.y) / current.zoom
        const pan =
          nextZoom === 1
            ? { x: 0, y: 0 }
            : clampPan(cursorX - box.left - worldX * nextZoom, cursorY - box.top - worldY * nextZoom, nextZoom, box, stageRect.width, stageRect.height)
        const next = { zoom: nextZoom, x: pan.x, y: pan.y }
        cameraRef.current = next
        setCamera(next)
        return
      }
      if (current.zoom <= 1) return
      event.preventDefault()
      const dx = event.shiftKey ? event.deltaY : event.deltaX
      const dy = event.shiftKey ? 0 : event.deltaY
      const pan = clampPan(current.x - dx, current.y - dy, current.zoom, box, stageRect.width, stageRect.height)
      const next = { zoom: current.zoom, x: pan.x, y: pan.y }
      cameraRef.current = next
      setCamera(next)
    }
    node.addEventListener('wheel', onWheel, { passive: false })
    return () => node.removeEventListener('wheel', onWheel)
  }, [])

  function measure() {
    const img = imgRef.current
    if (!img?.naturalWidth) return
    const rect = img.getBoundingClientRect()
    const parent = img.parentElement?.getBoundingClientRect()
    if (!parent) return
    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight)
    const width = img.naturalWidth * scale
    const height = img.naturalHeight * scale
    setNatural({ width: img.naturalWidth, height: img.naturalHeight })
    setFrame({
      left: rect.left - parent.left + (rect.width - width) / 2,
      top: rect.top - parent.top + (rect.height - height) / 2,
      width,
      height,
    })
  }

  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    measure()
    const observer = new ResizeObserver(() => measure())
    observer.observe(img)
    return () => observer.disconnect()
  }, [imageUrl])

  function eventPoint(event: { clientX: number; clientY: number; currentTarget: EventTarget & Element }): MarkPoint | null {
    const rect = event.currentTarget.getBoundingClientRect()
    if (!rect.width || !rect.height) return null
    return {
      x: clamp01((event.clientX - rect.left) / rect.width),
      y: clamp01((event.clientY - rect.top) / rect.height),
    }
  }

  function nudge(point: MarkPoint, size: FrameSize): MarkPoint {
    return clampPoint({ x: point.x + 0.6 / Math.max(size.width, 1), y: point.y })
  }

  function paintBrushLayer() {
    const canvas = brushCanvasRef.current
    const box = frameRef.current
    if (!canvas || box.width <= 0 || box.height <= 0) return
    const zoom = cameraRef.current.zoom
    const dpr = window.devicePixelRatio || 1
    const bitmapWidth = Math.max(1, Math.round(box.width * zoom * dpr))
    const bitmapHeight = Math.max(1, Math.round(box.height * zoom * dpr))
    if (canvas.width !== bitmapWidth || canvas.height !== bitmapHeight) {
      canvas.width = bitmapWidth
      canvas.height = bitmapHeight
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr * zoom, 0, 0, dpr * zoom, 0, 0)
    ctx.clearRect(0, 0, box.width, box.height)
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    const trace = (points: MarkPoint[]) => {
      ctx.beginPath()
      ctx.moveTo(points[0].x * box.width, points[0].y * box.height)
      if (points.length === 2) {
        ctx.lineTo(points[1].x * box.width, points[1].y * box.height)
      } else {
        for (let index = 1; index < points.length - 1; index += 1) {
          const current = points[index]
          const next = points[index + 1]
          ctx.quadraticCurveTo(
            current.x * box.width,
            current.y * box.height,
            ((current.x + next.x) / 2) * box.width,
            ((current.y + next.y) / 2) * box.height,
          )
        }
        const last = points[points.length - 1]
        ctx.lineTo(last.x * box.width, last.y * box.height)
      }
      ctx.stroke()
    }
    const paintStroke = (points: MarkPoint[], color: string, size: number, alpha: number, outline: boolean) => {
      if (points.length < 2) return
      if (outline) {
        ctx.globalAlpha = 1
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = size + 4
        trace(points)
      }
      ctx.globalAlpha = alpha
      ctx.strokeStyle = color
      ctx.lineWidth = size
      trace(points)
      ctx.globalAlpha = 1
    }

    const alpha = opacityRef.current
    for (const mark of marksRef.current) {
      if (mark.tool !== 'brush') continue
      paintStroke(mark.points, mark.color, mark.size ?? 18, Math.max(alpha, 0.35), mark.id === selectedRef.current)
    }
    const current = gestureRef.current
    if (current?.kind === 'move') {
      const ghost = marksRef.current.find((mark) => mark.id === current.id && mark.tool === 'brush')
      if (ghost) paintStroke(current.base, ghost.color, ghost.size ?? 18, Math.min(alpha, 0.18), false)
    }
    if (current?.kind === 'brush' && current.moved) {
      paintStroke(current.points, labelRef.current?.color ?? '#ff6d2d', brushRef.current, Math.max(alpha, 0.65), false)
    }
  }

  function selectedWord() {
    return (labelClass?.name ?? '').trim()
  }

  async function sendDetect(point?: MarkPoint | null) {
    const word = (labelRef.current?.name ?? '').trim()
    if (detectBusyRef.current) return
    if (!word) {
      setDetectFailed(true)
      setDetectNote('라벨 이름을 먼저 선택해 주세요.')
      return
    }
    if (!modelId) return
    if (!point) {
      setDetectFailed(true)
      setDetectNote('이미지 위를 클릭해 주세요.')
      return
    }
    detectBusyRef.current = true
    setDetectBusy(true)
    setDetectNote('')
    setDetectFailed(false)
    try {
      await detectApi({
        model: modelId,
        name: modelName,
        image: imageName,
        path: folderPath,
        tool,
        mode: detectModeRef.current,
        words: [word],
        word,
        x: point.x,
        y: point.y,
        pixelX: point.x < 0 ? -1 : natural.width > 0 ? Math.round(point.x * natural.width) : undefined,
        pixelY: point.y < 0 ? -1 : natural.height > 0 ? Math.round(point.y * natural.height) : undefined,
        width: natural.width || undefined,
        height: natural.height || undefined,
      }, imageFile)
      setDetectNote('전송했습니다.')
    } catch (error) {
      setDetectFailed(true)
      setDetectNote(
        error instanceof Error && error.message === '검출 API가 실패했습니다.'
          ? error.message
          : '백엔드에 연결하지 못했습니다. backend를 실행한 뒤 다시 검출해 주세요.',
      )
    } finally {
      detectBusyRef.current = false
      setDetectBusy(false)
    }
  }

  function onDetect() {
    if (detectBusy) return
    if (detectModeRef.current === 'all') {
      void sendDetect({ x: -1, y: -1 })
      return
    }
    void sendDetect(pointerRef.current)
  }

  function onPointerDown(event: ReactPointerEvent<SVGSVGElement>) {
    if (event.button !== 0) return
    if (performance.now() - closedAt.current < 280) return
    const point = eventPoint(event)
    if (!point) return
    if (event.altKey) event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    setPointer(point)
    pointerRef.current = point
    if (detectModeRef.current === 'click') {
      void sendDetect(point)
      return
    }
    const size = frameRef.current
    const marksNow = marksRef.current
    const activeTool = toolRef.current

    if (activeTool === 'polygon' && draftRef.current.length > 0) {
      const points = draftRef.current
      if (points.length >= 3 && pixelDistance(point, points[0], size) <= 12) {
        finishPolygon(points)
        return
      }
      replaceDraft([...points, point])
      return
    }

    if (activeTool === 'polygon') {
      const selected = marksNow.find((mark) => mark.id === selectedRef.current && mark.tool === 'polygon')
      if (selected && !event.altKey) {
        const vertex = hitVertex(selected.points, point, size)
        if (vertex >= 0) {
          beginHistory()
          setGestureNow({ kind: 'vertex', id: selected.id, index: vertex })
          return
        }
      }
      if (!event.altKey) {
        const hit = hitMark(marksNow, point, size)
        if (hit) {
          select(hit.id)
          beginHistory()
          setGestureNow({ kind: 'move', id: hit.id, origin: point, base: hit.points.map((item) => ({ ...item })) })
          return
        }
      }
      if (!labelRef.current?.name) return
      replaceDraft([point])
      select(null)
      return
    }

    if (activeTool === 'bbox') {
      const selected = marksNow.find((mark) => mark.id === selectedRef.current)
      if (selected?.tool === 'bbox' && !event.altKey) {
        const box = boxOf(selected.points)
        const handle = box ? hitBoxHandle(box, point, size) : null
        if (box && handle) {
          beginHistory()
          setGestureNow({ kind: 'resize', id: selected.id, handle, box })
          return
        }
      }
      if (!event.altKey) {
        const hit = hitMark(marksNow, point, size)
        if (hit) {
          select(hit.id)
          beginHistory()
          setGestureNow({ kind: 'move', id: hit.id, origin: point, base: hit.points.map((item) => ({ ...item })) })
          return
        }
      }
      if (!labelRef.current?.name) return
      select(null)
      replaceDraft([point, point])
      setGestureNow({ kind: 'bbox', start: point })
      return
    }

    if (eraseRef.current) {
      beginHistory()
      setGestureNow({ kind: 'erase', last: point })
      const erased = eraseBrush(marksRef.current, point, brushRef.current / 2, size)
      if (JSON.stringify(erased) !== JSON.stringify(marksRef.current)) setMarksNow(erased)
      return
    }
    const selectedBrush = marksNow.find((mark) => mark.id === selectedRef.current && mark.tool === 'brush')
    if (selectedBrush && !event.altKey && hitMark([selectedBrush], point, size)) {
      beginHistory()
      setGestureNow({ kind: 'move', id: selectedBrush.id, origin: point, base: selectedBrush.points.map((item) => ({ ...item })) })
      return
    }
    setGestureNow({ kind: 'brush', points: [point], moved: false })
  }

  function onPointerMove(event: ReactPointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    if (!rect.width || !rect.height) return
    const native = event.nativeEvent
    const coalesced = native.getCoalescedEvents?.() ?? []
    const samples = (coalesced.length > 0 ? coalesced : [native]).map((item) => ({
      x: clamp01((item.clientX - rect.left) / rect.width),
      y: clamp01((item.clientY - rect.top) / rect.height),
    }))
    const point = samples[samples.length - 1]
    if (!point) return
    setPointer(point)
    const current = gestureRef.current
    const size = frameRef.current
    if (!current) return

    if (current.kind === 'bbox') {
      const end = event.shiftKey ? squarePoint(current.start, point, size) : point
      replaceDraft([current.start, end])
      return
    }
    if (current.kind === 'brush') {
      const points = current.points.slice()
      let moved = current.moved
      for (const sample of samples) {
        const last = points[points.length - 1]
        const distance = pixelDistance(last, sample, size)
        if (distance < 0.5) continue
        if (distance >= 1) moved = true
        const steps = Math.max(1, Math.ceil(distance / 2))
        for (let step = 1; step <= steps; step += 1) {
          const t = step / steps
          points.push({
            x: last.x + (sample.x - last.x) * t,
            y: last.y + (sample.y - last.y) * t,
          })
        }
      }
      gestureRef.current = { ...current, points, moved }
      paintBrushLayer()
      return
    }
    if (current.kind === 'erase') {
      let erased = marksRef.current
      let last = current.last
      for (const sample of samples) {
        const steps = Math.max(1, Math.ceil(pixelDistance(last, sample, size) / Math.max(2, brushRef.current / 4)))
        for (let step = 1; step <= steps; step += 1) {
          const t = step / steps
          erased = eraseBrush(
            erased,
            { x: last.x + (sample.x - last.x) * t, y: last.y + (sample.y - last.y) * t },
            brushRef.current / 2,
            size,
          )
        }
        last = sample
      }
      gestureRef.current = { ...current, last: point }
      if (JSON.stringify(erased) !== JSON.stringify(marksRef.current)) setMarksNow(erased)
      return
    }
    if (current.kind === 'move') {
      const moved = translateInside(current.base, point.x - current.origin.x, point.y - current.origin.y)
      setMarksNow(marksRef.current.map((mark) => (mark.id === current.id ? { ...mark, points: moved } : mark)))
      return
    }
    if (current.kind === 'resize') {
      const box = resizeBox(current.box, current.handle, point, size, event.shiftKey)
      setMarksNow(marksRef.current.map((mark) => (mark.id === current.id ? { ...mark, points: boxPoints(box) } : mark)))
      return
    }
    const clamped = clampPoint(point)
    setMarksNow(
      marksRef.current.map((mark) =>
        mark.id === current.id ? { ...mark, points: mark.points.map((item, index) => (index === current.index ? clamped : item)) } : mark,
      ),
    )
  }

  function onPointerUp(event: ReactPointerEvent<SVGSVGElement>) {
    const current = gestureRef.current
    if (!current) return
    gestureRef.current = null
    setGesture(null)
    const point = eventPoint(event)
    const size = frameRef.current

    if (current.kind === 'bbox') {
      const end = point ?? current.start
      const corner = event.shiftKey ? squarePoint(current.start, end, size) : end
      replaceDraft([])
      const box = boxOf([current.start, corner])
      if (!box || (box.w * size.width < 4 && box.h * size.height < 4)) {
        select(null)
        return
      }
      finishMark('bbox', boxPoints(box))
      return
    }
    if (current.kind === 'brush') {
      replaceDraft([])
      if (!current.moved) {
        const hit = hitMark(marksRef.current, current.points[0], size)
        if (hit) {
          select(hit.id)
          return
        }
        finishMark('brush', [current.points[0], nudge(current.points[0], size)], brushRef.current)
        return
      }
      finishMark('brush', current.points, brushRef.current)
      return
    }
    commitHistory()
  }

  function onContextMenu(event: MouseEvent<SVGSVGElement>) {
    event.preventDefault()
    if (detectModeRef.current === 'click') return
    if (tool === 'polygon' && draftRef.current.length > 0) {
      replaceDraft(draftRef.current.slice(0, -1))
      return
    }
    const point = eventPoint(event)
    if (!point || tool !== 'polygon') return
    const selected = marksRef.current.find((mark) => mark.id === selectedRef.current && mark.tool === 'polygon')
    if (!selected || selected.points.length <= 3) return
    const index = hitVertex(selected.points, point, frame)
    if (index < 0) return
    const before = marksRef.current
    setMarksNow(before.map((mark) => (mark.id === selected.id ? { ...mark, points: mark.points.filter((_, item) => item !== index) } : mark)))
    pushPast(before)
  }

  function onDoubleClick() {
    if (detectModeRef.current === 'click' || tool !== 'polygon') return
    const points = draftRef.current
    if (points.length >= 4) finishPolygon(points.slice(0, -1))
    else if (points.length >= 3) finishPolygon(points)
  }

  const draftColor = labelClass?.color ?? '#ff6d2d'
  const classIndex = labelClass ? Math.max(0, classes.indexOf(labelClass)) : 0
  const selected = marks.find((mark) => mark.id === selectedId) ?? null
  const closing =
    tool === 'polygon' && draft.length >= 3 && pointer !== null && frame.width > 0 && pixelDistance(pointer, draft[0], frame) <= 12
  const polygonPreview = tool === 'polygon' && draft.length > 0 ? (pointer && !closing ? [...draft, pointer] : draft) : []
  const preview: Mark | null =
    tool === 'bbox' && gesture?.kind === 'bbox' && draft.length >= 2
      ? { id: 'draft', tool: 'bbox', name: labelClass?.name ?? '', color: draftColor, points: draft }
      : polygonPreview.length >= 2
        ? { id: 'draft', tool: 'polygon', name: labelClass?.name ?? '', color: draftColor, points: polygonPreview }
        : null
  const selectedBox = selected?.tool === 'bbox' ? boxOf(selected.points) : null
  const hoverHandle =
    tool === 'bbox' && selectedBox && pointer && gesture?.kind !== 'bbox' ? hitBoxHandle(selectedBox, pointer, frame) : null
  const drawing = gesture?.kind === 'bbox' || (tool === 'polygon' && draft.length > 0)
  let cursor = detectMode === 'click' || tool !== 'brush' ? 'crosshair' : 'none'
  if (detectMode !== 'click' && gesture?.kind === 'resize') cursor = RESIZE_CURSOR[gesture.handle]
  else if (detectMode !== 'click' && (gesture?.kind === 'move' || gesture?.kind === 'vertex')) cursor = 'grabbing'
  else if (detectMode !== 'click' && !drawing && hoverHandle) cursor = RESIZE_CURSOR[hoverHandle]
  else if (detectMode !== 'click' && !drawing && tool !== 'brush' && pointer && hitMark(marks, pointer, frame)) cursor = 'move'

  const sizeBox =
    gesture?.kind === 'bbox'
      ? boxOf(draft)
      : gesture?.kind === 'resize' || (gesture?.kind === 'move' && selected?.tool === 'bbox')
        ? selectedBox
        : null
  const ghost =
    gesture?.kind === 'move'
      ? marks.find((mark) => mark.id === gesture.id)
      : gesture?.kind === 'resize'
        ? marks.find((mark) => mark.id === gesture.id)
        : null
  const vertices =
    tool === 'polygon' ? (draft.length > 0 ? draft : selected?.tool === 'polygon' ? selected.points : []) : []
  const layerStyle = { left: 0, top: 0, width: frame.width, height: frame.height }
  const worldStyle = {
    left: frame.left,
    top: frame.top,
    width: frame.width,
    height: frame.height,
    transform: `translate(${camera.x}px, ${camera.y}px) scale(${camera.zoom})`,
  }
  const px = (value: number) => value / (frame.width * camera.zoom)
  const py = (value: number) => value / (frame.height * camera.zoom)

  return (
    <div className="label-workspace">
      <div className="label-bar">
        <div className="seg" role="radiogroup" aria-label="라벨 도구">
          {TOOLS.map((item) => (
            <button
              key={item.id}
              type="button"
              role="radio"
              aria-checked={tool === item.id}
              title={`${item.label} (${item.key})`}
              disabled={detectMode === 'click'}
              onClick={() => onTool(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="class-pick" ref={labelMenuRef}>
          <span className="detect-label">라벨</span>
          {classes.length === 0 ? (
            <p className="detect-note">종류를 먼저 추가해 주세요</p>
          ) : (
            <>
              <button
                type="button"
                className="class-select"
                aria-haspopup="listbox"
                aria-expanded={labelOpen}
                onClick={() => {
                  const next = !labelOpenRef.current
                  labelOpenRef.current = next
                  setLabelOpen(next)
                }}
              >
                <i style={{ background: labelClass?.color ?? classes[classIndex]?.color ?? '#ff6d2d' }} />
                <span>{(labelClass ?? classes[classIndex])?.name.trim() || '이름 없음'}</span>
              </button>
              {labelOpen ? (
                <div className="class-menu" role="listbox" aria-label="라벨 이름">
                  {classes.map((item, index) => (
                    <button
                      key={index}
                      type="button"
                      role="option"
                      aria-selected={index === classIndex}
                      onClick={() => {
                        onClassIndex(index)
                        labelOpenRef.current = false
                        setLabelOpen(false)
                      }}
                    >
                      <i style={{ background: item.color }} />
                      <span>{item.name.trim() || '이름 없음'}</span>
                    </button>
                  ))}
                </div>
              ) : null}
            </>
          )}
        </div>
        {tool === 'brush' ? (
          <>
            <label className="brush-size">
              크기 {brushSize}
              <input
                className="slider"
                type="range"
                min={4}
                max={72}
                value={brushSize}
                style={{ ['--fill' as string]: `${((brushSize - 4) / 68) * 100}%` }}
                onChange={(event) => changeBrush(Number(event.target.value))}
                aria-label="브러시 크기"
                disabled={detectMode === 'click'}
              />
            </label>
            <button type="button" className={erasing ? 'mini active' : 'mini'} aria-pressed={erasing} onClick={toggleErase} disabled={detectMode === 'click'}>
              지우개
            </button>
          </>
        ) : null}
        <button type="button" className="mini" onClick={undo} disabled={detectMode === 'click'}>
          되돌리기
        </button>
        <button type="button" className="mini" onClick={redo} disabled={detectMode === 'click'}>
          다시
        </button>
        <div className="detect-block">
          <span className="detect-label">검출</span>
          <div className="seg" role="radiogroup" aria-label="검출 범위">
            <button
              type="button"
              role="radio"
              aria-checked={detectMode === 'all'}
              onClick={() => onDetectMode('all')}
            >
              전체
            </button>
            <button
              type="button"
              role="radio"
              aria-checked={detectMode === 'click'}
              onClick={() => onDetectMode('click')}
            >
              클릭
            </button>
          </div>
          <button
            type="button"
            className="mini"
            onClick={onDetect}
            disabled={detectBusy || !modelId || selectedWord().length === 0}
          >
            {detectBusy ? '검출 중…' : '검출'}
          </button>
          {detectNote ? <p className={detectFailed ? 'detect-note failed' : 'detect-note'}>{detectNote}</p> : null}
        </div>
      </div>
      <div className="label-main">
      <div className="stage" ref={stageRef}>
        <img
          ref={imgRef}
          className="stage-fit"
          src={imageUrl}
          alt=""
          draggable={false}
          onLoad={measure}
          style={{ visibility: frame.width > 0 ? 'hidden' : 'visible' }}
        />
        {frame.width > 0 ? (
          <div className="stage-world" style={worldStyle}>
            <img src={imageUrl} alt={imageName} draggable={false} />
            <canvas ref={brushCanvasRef} className="brush-layer" style={layerStyle} />
            <svg
              viewBox="0 0 1 1"
              preserveAspectRatio="none"
              style={{ ...layerStyle, cursor }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onLostPointerCapture={onPointerUp}
              onPointerLeave={() => {
                if (!gestureRef.current) setPointer(null)
              }}
              onContextMenu={onContextMenu}
              onDoubleClick={onDoubleClick}
            >
              <rect x={0} y={0} width={1} height={1} fill="#ffffff" fillOpacity={0} pointerEvents="all" />
              {ghost && gesture?.kind === 'move' ? (
                <MarkShape
                  mark={{ ...ghost, id: 'ghost', points: gesture.base }}
                  opacity={Math.min(opacity, 0.18)}
                  selected={false}
                  preview
                />
              ) : null}
              {ghost && gesture?.kind === 'resize' ? (
                <MarkShape
                  mark={{ ...ghost, id: 'ghost', points: boxPoints(gesture.box) }}
                  opacity={Math.min(opacity, 0.18)}
                  selected={false}
                  preview
                />
              ) : null}
              {marks.map((mark) => (
                <MarkShape key={mark.id} mark={mark} opacity={opacity} selected={mark.id === selectedId} preview={false} />
              ))}
              {preview ? <MarkShape mark={preview} opacity={opacity} selected={false} preview /> : null}
              {gesture?.kind === 'bbox' && pointer ? (
                <>
                  <line x1={pointer.x} y1={0} x2={pointer.x} y2={1} stroke="#ffffff" strokeOpacity={0.55} strokeWidth={1} vectorEffect="non-scaling-stroke" />
                  <line x1={0} y1={pointer.y} x2={1} y2={pointer.y} stroke="#ffffff" strokeOpacity={0.55} strokeWidth={1} vectorEffect="non-scaling-stroke" />
                </>
              ) : null}
              {tool === 'bbox' && !gesture && pointer ? (
                <>
                  <line
                    x1={pointer.x - px(8)}
                    x2={pointer.x + px(8)}
                    y1={pointer.y}
                    y2={pointer.y}
                    stroke="#ffffff"
                    strokeOpacity={0.8}
                    strokeWidth={1}
                    vectorEffect="non-scaling-stroke"
                  />
                  <line
                    x1={pointer.x}
                    x2={pointer.x}
                    y1={pointer.y - py(8)}
                    y2={pointer.y + py(8)}
                    stroke="#ffffff"
                    strokeOpacity={0.8}
                    strokeWidth={1}
                    vectorEffect="non-scaling-stroke"
                  />
                </>
              ) : null}
              {vertices.map((point, index) => (
                <ellipse
                  key={index}
                  cx={point.x}
                  cy={point.y}
                  rx={px(index === 0 && closing ? 8 : 5)}
                  ry={py(index === 0 && closing ? 8 : 5)}
                  fill={index === 0 && closing ? '#ffffff' : draftColor}
                  stroke="#111111"
                  strokeWidth={1}
                  vectorEffect="non-scaling-stroke"
                />
              ))}
              {tool === 'bbox' && selected?.tool === 'bbox' && selectedBox
                ? BOX_HANDLES.map((handle) => {
                    const point = handlePoint(selectedBox, handle)
                    const hot = hoverHandle === handle || (gesture?.kind === 'resize' && gesture.handle === handle)
                    return (
                      <ellipse
                        key={handle}
                        cx={point.x}
                        cy={point.y}
                        rx={px(5)}
                        ry={py(5)}
                        fill={hot ? selected.color : '#ffffff'}
                        stroke="#111111"
                        strokeWidth={1}
                        vectorEffect="non-scaling-stroke"
                      />
                    )
                  })
                : null}
            </svg>
            <div className="stage-ui" style={layerStyle}>
              {marks.map((mark) => {
                const anchor = labelAnchor(mark)
                if (!anchor || !mark.name) return null
                return (
                  <span
                    key={mark.id}
                    className="mark-tag"
                    style={{
                      left: `${anchor.x * 100}%`,
                      top: `${anchor.y * 100}%`,
                      background: mark.color,
                      transform: anchor.y < 0.06 ? 'translateY(2px)' : 'translateY(calc(-100% - 2px))',
                    }}
                  >
                    {mark.name}
                  </span>
                )
              })}
              {sizeBox && natural.width > 0 && (sizeBox.w > 0 || sizeBox.h > 0) ? (
                <span
                  className="size-tag"
                  style={{ left: `${(sizeBox.x + sizeBox.w) * 100}%`, top: `${(sizeBox.y + sizeBox.h) * 100}%` }}
                >
                  {Math.max(1, Math.round(sizeBox.w * natural.width))} × {Math.max(1, Math.round(sizeBox.h * natural.height))}
                </span>
              ) : null}
              {tool === 'brush' && pointer ? (
                <span
                  className={erasing ? 'brush-cursor erasing' : 'brush-cursor'}
                  style={{
                    left: pointer.x * frame.width,
                    top: pointer.y * frame.height,
                    width: brushSize,
                    height: brushSize,
                    borderColor: erasing ? '#ffb4b4' : draftColor,
                  }}
                />
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
      {marks.length > 0 ? (
        <div className="mark-list">
          {marks.map((mark, index) => (
            <div key={mark.id} className={mark.id === selectedId ? 'mark-chip active' : 'mark-chip'}>
              <button type="button" onClick={() => select(mark.id)}>
                <i style={{ background: mark.color }} />
                {mark.name || 'label'} {index + 1}
              </button>
              <button type="button" aria-label="삭제" onClick={() => removeMark(mark.id)}>
                ×
              </button>
            </div>
          ))}
        </div>
      ) : null}
      </div>
    </div>
  )
}

function MarkShape({ mark, opacity, selected, preview }: { mark: Mark; opacity: number; selected: boolean; preview: boolean }) {
  const stroke = preview ? '#ffffff' : selected ? '#ffffff' : mark.color
  const dash = preview ? '6 4' : undefined
  if (mark.tool === 'bbox' && mark.points.length >= 2) {
    const box = boxOf(mark.points)
    if (!box) return null
    return (
      <rect
        x={box.x}
        y={box.y}
        width={box.w}
        height={box.h}
        fill={mark.color}
        fillOpacity={preview ? Math.max(opacity, 0.28) : opacity}
        stroke={stroke}
        strokeWidth={selected ? 2.5 : 2}
        strokeDasharray={dash}
        vectorEffect="non-scaling-stroke"
      />
    )
  }
  if (mark.tool === 'polygon' && mark.points.length >= 2) {
    const points = mark.points.map((point) => `${point.x},${point.y}`).join(' ')
    return (
      <polygon
        points={points}
        fill={mark.color}
        fillOpacity={mark.points.length >= 3 ? (preview ? Math.max(opacity, 0.28) : opacity) : 0}
        stroke={stroke}
        strokeWidth={selected ? 2.5 : 2}
        strokeDasharray={dash}
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    )
  }
  return null
}
