import { useRef, useState, type ChangeEvent } from 'react'
import type { ImageEntry } from './folder'
import { labeledImages } from './marks'
import type { Project } from './projects'
import { RFDETR_SIZES, startTrainApi, uploadModelApi, type RfdetrSize } from './rfdetr'

type Tab = 'train' | 'infer'

function onnxFile(file: File) {
  return file.name.toLowerCase().endsWith('.onnx')
}

function errorText(error: unknown, fallback: string) {
  if (error instanceof TypeError) return '백엔드에 연결하지 못했습니다. backend를 실행한 뒤 다시 시도해 주세요.'
  return error instanceof Error && error.message ? error.message : fallback
}

export function RfdetrPanel({
  project,
  images,
}: {
  project: Project | null
  images: ImageEntry[]
  onLabeled: (imageNames: string[]) => void
}) {
  const [tab, setTab] = useState<Tab>('train')
  const [size, setSize] = useState<RfdetrSize>('small')
  const [epochs, setEpochs] = useState(50)
  const [batchSize, setBatchSize] = useState(4)
  const [training, setTraining] = useState(false)
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [modelReady, setModelReady] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [threshold, setThreshold] = useState(0.5)
  const [note, setNote] = useState('')
  const [failed, setFailed] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const projectId = project?.id ?? ''
  const labeled = projectId ? labeledImages(projectId) : new Set<string>()
  const labeledCount = images.filter((image) => labeled.has(image.name)).length

  function report(message: string, isError = false) {
    setNote(message)
    setFailed(isError)
  }

  async function onTrain() {
    if (!project || training) return
    report('')
    setTraining(true)
    try {
      await startTrainApi({
        projectId: project.id,
        size,
        epochs,
        batchSize,
      })
      report('학습 요청을 보냈습니다.')
    } catch (error) {
      report(errorText(error, '학습을 시작하지 못했습니다.'), true)
    } finally {
      setTraining(false)
    }
  }

  function onPickFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null
    event.target.value = ''
    if (!file) return
    if (!onnxFile(file)) {
      setModelFile(null)
      setModelReady(false)
      report('ONNX 파일만 선택할 수 있습니다.', true)
      return
    }
    setModelFile(file)
    setModelReady(false)
    report('')
  }

  async function onLoadModel() {
    if (!modelFile || !onnxFile(modelFile) || uploading) return
    report('')
    setUploading(true)
    try {
      await uploadModelApi(modelFile)
      setModelReady(true)
      report('모델을 불러왔습니다.')
    } catch (error) {
      setModelReady(false)
      report(errorText(error, '모델을 불러오지 못했습니다.'), true)
    } finally {
      setUploading(false)
    }
  }

  if (!project) {
    return <p className="model-status">프로젝트를 열면 학습하고 추론할 수 있습니다.</p>
  }

  return (
    <div className="rfdetr">
      <div className="pill-seg" role="tablist" aria-label="RF-DETR 작업">
        <button type="button" role="tab" aria-selected={tab === 'train'} onClick={() => setTab('train')}>
          학습
        </button>
        <button type="button" role="tab" aria-selected={tab === 'infer'} onClick={() => setTab('infer')}>
          추론
        </button>
      </div>

      {tab === 'train' && images.length === 0 ? (
        <p className="model-status">이미지 폴더를 먼저 열어 주세요.</p>
      ) : tab === 'train' ? (
        <div className="rfdetr-body">
          <dl className="rfdetr-stats">
            <div>
              <dt>라벨된 이미지</dt>
              <dd>
                {labeledCount} / {images.length}
              </dd>
            </div>
            <div>
              <dt>클래스</dt>
              <dd>{project.classes.length}개</dd>
            </div>
          </dl>

          <span className="field-label">크기</span>
          <div className="pill-seg" role="radiogroup" aria-label="모델 크기">
            {RFDETR_SIZES.map((item) => (
              <button
                key={item.id}
                type="button"
                role="radio"
                aria-checked={size === item.id}
                onClick={() => setSize(item.id)}
                disabled={training}
              >
                {item.label}
              </button>
            ))}
          </div>

          <label className="field-label rfdetr-epochs">
            epoch
            <input
              type="number"
              min={1}
              max={500}
              value={epochs}
              onChange={(event) => setEpochs(Math.min(500, Math.max(1, Number(event.target.value) || 1)))}
              disabled={training}
            />
          </label>

          <label className="field-label rfdetr-epochs">
            batch size
            <input
              type="number"
              min={1}
              max={128}
              value={batchSize}
              onChange={(event) => setBatchSize(Math.min(128, Math.max(1, Number(event.target.value) || 1)))}
              disabled={training}
            />
          </label>

          <button type="button" className="primary" onClick={onTrain} disabled={labeledCount === 0 || training}>
            학습 시작
          </button>
          {labeledCount === 0 ? <p className="model-status">라벨된 이미지가 있어야 학습할 수 있습니다.</p> : null}
        </div>
      ) : (
        <div className="rfdetr-body">
          <span className="field-label">ONNX 파일</span>
          <input ref={fileRef} type="file" accept=".onnx" hidden onChange={onPickFile} />
          <button type="button" className="ghost" onClick={() => fileRef.current?.click()}>
            파일 선택
          </button>
          <p className="rfdetr-file" title={modelFile?.name ?? ''}>
            {modelFile ? modelFile.name : '선택된 파일 없음'}
          </p>
          <button type="button" className="primary" onClick={onLoadModel} disabled={!modelFile || uploading}>
            불러오기
          </button>
          {modelReady ? <p className="model-status ready">불러옴</p> : null}

          <label className="field-label">
            신뢰도 {threshold.toFixed(2)}
            <input
              className="slider"
              type="range"
              min={5}
              max={95}
              step={5}
              style={{ ['--fill' as string]: `${((threshold * 100 - 5) / 90) * 100}%` }}
              value={Math.round(threshold * 100)}
              onChange={(event) => setThreshold(Number(event.target.value) / 100)}
              aria-label="신뢰도"
            />
          </label>
        </div>
      )}

      {note ? <p className={failed ? 'error' : 'model-status'}>{note}</p> : null}
    </div>
  )
}
