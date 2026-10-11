import { useEffect, useRef, useState, type PointerEvent } from 'react'
import { hitMark, loadMarks, pointNear, saveMarks, type Mark, type MarkPoint } from './marks'
import type { LabelClass, LabelTool } from './projects'

type Frame = { left: number; top: number; width: number; height: number }

const HINTS: Record<LabelTool, string> = {
  bbox: '빈 곳을 드래그하면 사각형이 생깁니다.',
  polygon: '점을 찍고, 시작점을 다시 누르면 닫힙니다.',
  brush: '드래그하면 칠해집니다.',
}

export function LabelStage({
  projectId,
  imageName,
  imageUrl,
  tool,
  opacity,
  labelClass,
}: {
  projectId: string
  imageName: string
  imageUrl: string
  tool: LabelTool
  opacity: number
  labelClass: LabelClass | null
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const draftRef = useRef<MarkPoint[]>([])
  const dragRef = useRef<MarkPoint | null>(null)
  const [frame, setFrame] = useState<Frame>({ left: 0, top: 0, width: 0, height: 0 })
  const [marks, setMarks] = useState<Mark[]>(() => loadMarks(projectId, imageName))
  const [draft, setDraft] = useState<MarkPoint[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)

  function replaceDraft(points: MarkPoint[]) {
    draftRef.current = points
    setDraft(points)
  }

  useEffect(() => {
    saveMarks(projectId, imageName, marks)
  }, [projectId, imageName, marks])

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return
      if (event.key === 'Escape') replaceDraft([])
      if ((event.key === 'Delete' || event.key === 'Backspace') && selectedId) {
        event.preventDefault()
        setMarks((list) => list.filter((mark) => mark.id !== selectedId))
        setSelectedId(null)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedId])

  function measure() {
    const img = imgRef.current
    if (!img?.naturalWidth) return
    const rect = img.getBoundingClientRect()
    const parent = img.parentElement?.getBoundingClientRect()
    if (!parent) return
    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight)
    const width = img.naturalWidth * scale
    const height = img.naturalHeight * scale
    setFrame({
      left: rect.left - parent.left + (rect.width - width) / 2,
      top: rect.top - parent.top + (rect.height - height) / 2,
      width,
      height,
    })
  }

  useEffect(() => {
    draftRef.current = []
    setDraft([])
  }, [tool])

  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    measure()
    const observer = new ResizeObserver(() => measure())
    observer.observe(img)
    return () => observer.disconnect()
  }, [imageUrl])

  function eventPoint(event: PointerEvent<SVGSVGElement>): MarkPoint | null {
    const rect = event.currentTarget.getBoundingClientRect()
    if (!rect.width || !rect.height) return null
    const x = (event.clientX - rect.left) / rect.width
    const y = (event.clientY - rect.top) / rect.height
    if (x < 0 || y < 0 || x > 1 || y > 1) return null
    return { x, y }
  }

  function commit(points: MarkPoint[]) {
    if (!labelClass?.name) return
    setMarks((list) => [
      ...list,
      { id: crypto.randomUUID(), tool, name: labelClass.name, color: labelClass.color, points },
    ])
    replaceDraft([])
    setSelectedId(null)
  }

  function onPointerDown(event: PointerEvent<SVGSVGElement>) {
    if (!labelClass?.name) return
    const point = eventPoint(event)
    if (!point) return
    event.currentTarget.setPointerCapture(event.pointerId)
    if (tool === 'polygon') {
      if (draftRef.current.length >= 3 && pointNear(point, draftRef.current[0])) {
        commit(draftRef.current)
        return
      }
      if (draftRef.current.length === 0) {
        const found = hitMark(marks, point)
        if (found) {
          setSelectedId(found.id)
          return
        }
      }
      replaceDraft([...draftRef.current, point])
      return
    }
    dragRef.current = point
    replaceDraft([point])
  }

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    if (tool === 'polygon' || !dragRef.current) return
    const point = eventPoint(event)
    if (!point) return
    if (tool === 'bbox') replaceDraft([dragRef.current, point])
    else replaceDraft([...draftRef.current, point])
  }

  function onPointerUp(event: PointerEvent<SVGSVGElement>) {
    if (tool === 'polygon') return
    const start = dragRef.current
    dragRef.current = null
    const point = eventPoint(event) ?? start
    if (!start || !point) return
    if (tool === 'bbox') {
      if (pointNear(start, point, 0.01)) {
        replaceDraft([])
        setSelectedId(hitMark(marks, start)?.id ?? null)
        return
      }
      commit([start, point])
      return
    }
    const stroke = draftRef.current.length > 1 ? draftRef.current : [start, point]
    const moved = stroke.some((item) => !pointNear(stroke[0], item, 0.01))
    if (!moved) {
      replaceDraft([])
      setSelectedId(hitMark(marks, start)?.id ?? null)
      return
    }
    commit(stroke)
  }

  const draftColor = labelClass?.color ?? '#ff6d2d'

  return (
    <>
      <div className="stage">
        <img ref={imgRef} src={imageUrl} alt={imageName} draggable={false} onLoad={measure} />
        {frame.width > 0 ? (
          <svg
            viewBox="0 0 1 1"
            preserveAspectRatio="none"
            style={{ left: frame.left, top: frame.top, width: frame.width, height: frame.height }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          >
            {marks.map((mark) => (
              <MarkShape key={mark.id} mark={mark} opacity={opacity} selected={mark.id === selectedId} />
            ))}
            {draft.length > 0 ? (
              <MarkShape
                mark={{ id: 'draft', tool, name: '', color: draftColor, points: draft }}
                opacity={opacity}
                selected
              />
            ) : null}
          </svg>
        ) : null}
      </div>
      <p className="viewer-hint">
        {labelClass?.name ? `${labelClass.name} · ${HINTS[tool]} 선택한 라벨은 Delete로 지웁니다.` : '객체 종류 이름을 적어 주세요.'}
      </p>
    </>
  )
}

function MarkShape({ mark, opacity, selected }: { mark: Mark; opacity: number; selected: boolean }) {
  const stroke = selected ? '#ffffff' : mark.color
  if (mark.tool === 'bbox' && mark.points.length >= 2) {
    const [a, b] = mark.points
    const x = Math.min(a.x, b.x)
    const y = Math.min(a.y, b.y)
    return (
      <rect
        x={x}
        y={y}
        width={Math.abs(b.x - a.x)}
        height={Math.abs(b.y - a.y)}
        fill={mark.color}
        fillOpacity={opacity}
        stroke={stroke}
        strokeWidth={2}
        vectorEffect="non-scaling-stroke"
      />
    )
  }
  if (mark.tool === 'polygon' && mark.points.length >= 2) {
    return (
      <polygon
        points={mark.points.map((point) => `${point.x},${point.y}`).join(' ')}
        fill={mark.color}
        fillOpacity={mark.points.length >= 3 ? opacity : 0}
        stroke={stroke}
        strokeWidth={2}
        vectorEffect="non-scaling-stroke"
      />
    )
  }
  if (mark.points.length >= 2) {
    return (
      <polyline
        points={mark.points.map((point) => `${point.x},${point.y}`).join(' ')}
        fill="none"
        stroke={mark.color}
        strokeOpacity={Math.max(opacity, 0.35)}
        strokeWidth={18}
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    )
  }
  return null
}
