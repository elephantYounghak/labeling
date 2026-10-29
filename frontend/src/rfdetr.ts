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
  batchSize: number
}

async function errorMessage(res: Response, fallback: string) {
  const body = await res.json().catch(() => null)
  const detail = body && typeof body.error === 'string' ? body.error : ''
  return detail || fallback
}

export async function uploadModelApi(file: File): Promise<void> {
  const form = new FormData()
  form.append('file', file, file.name)
  const res = await fetch('/ai/upload', { method: 'POST', body: form })
  if (!res.ok) throw new Error(await errorMessage(res, '모델 업로드 API가 실패했습니다.'))
}

export async function startTrainApi(request: TrainRequest): Promise<string> {
  const res = await fetch('/ai/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) throw new Error(await errorMessage(res, '학습 API가 실패했습니다.'))
  const saved = (await res.json().catch(() => null)) as { id?: unknown }
  return typeof saved?.id === 'string' ? saved.id : ''
}
