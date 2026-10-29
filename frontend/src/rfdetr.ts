import type { LabelClass } from './projects'

export type RfdetrSize = 'small' | 'medium' | 'large'

export const RFDETR_SIZES: { id: RfdetrSize; label: string }[] = [
  { id: 'small', label: 'Small' },
  { id: 'medium', label: 'Medium' },
  { id: 'large', label: 'Large' },
]

export type TrainRequest = {
  projectId: string
  size: RfdetrSize
  epochs: number
  images: number
  classes: LabelClass[]
}

export type TrainStatus = {
  state: 'running' | 'done' | 'stopped' | 'failed'
  epoch: number
  epochs: number
  map: number | null
  error?: string
}

export type RfdetrRun = {
  id: string
  size: RfdetrSize
  epochs: number
  images: number
  map: number
  createdAt: string
}

export type InferRequest = {
  projectId: string
  runId: string
  threshold: number
  file: File
}

// 백엔드 학습 API가 생기기 전까지 쓰는 가짜 구현이다. 함수 모양은 실제 API와 같게 유지한다.
const SECONDS_PER_EPOCH = 0.4

type FakeJob = {
  request: TrainRequest
  startedAt: number
  stoppedAt: number | null
  saved: boolean
}

const jobs = new Map<string, FakeJob>()

function runsKey(projectId: string) {
  return `labeling.rfdetr.runs.${projectId}`
}

function loadRuns(projectId: string): RfdetrRun[] {
  try {
    const raw = localStorage.getItem(runsKey(projectId))
    const parsed = raw ? (JSON.parse(raw) as RfdetrRun[]) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function fakeMap(epoch: number) {
  return Number((0.75 * (1 - Math.exp(-epoch / 12))).toFixed(3))
}

export async function listRunsApi(projectId: string): Promise<RfdetrRun[]> {
  return loadRuns(projectId)
}

export async function startTrainApi(request: TrainRequest): Promise<string> {
  const id = crypto.randomUUID()
  jobs.set(id, { request, startedAt: Date.now(), stoppedAt: null, saved: false })
  return id
}

export async function trainStatusApi(jobId: string): Promise<TrainStatus> {
  const job = jobs.get(jobId)
  if (!job) {
    return { state: 'failed', epoch: 0, epochs: 0, map: null, error: '학습 작업을 찾지 못했습니다.' }
  }
  const { epochs } = job.request
  const end = job.stoppedAt ?? Date.now()
  const epoch = Math.min(epochs, Math.floor((end - job.startedAt) / 1000 / SECONDS_PER_EPOCH))
  const map = epoch > 0 ? fakeMap(epoch) : null
  if (job.stoppedAt !== null) return { state: 'stopped', epoch, epochs, map }
  if (epoch < epochs) return { state: 'running', epoch, epochs, map }
  if (!job.saved) {
    job.saved = true
    const run: RfdetrRun = {
      id: jobId,
      size: job.request.size,
      epochs,
      images: job.request.images,
      map: fakeMap(epochs),
      createdAt: new Date().toISOString(),
    }
    const runs = [run, ...loadRuns(job.request.projectId)]
    localStorage.setItem(runsKey(job.request.projectId), JSON.stringify(runs))
  }
  return { state: 'done', epoch, epochs, map }
}

export async function stopTrainApi(jobId: string): Promise<void> {
  const job = jobs.get(jobId)
  if (job && job.stoppedAt === null) job.stoppedAt = Date.now()
}

export async function inferApi(request: InferRequest): Promise<unknown> {
  await new Promise((resolve) => setTimeout(resolve, 80))
  return { run: request.runId, boxes: [] }
}
