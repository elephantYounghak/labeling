export type AiModel = {
  id: string
  name: string
  files: string[]
}

const STORAGE_SELECTED = 'labeling.model'

export const MODELS: AiModel[] = [
  {
    id: 'sam3',
    name: 'SAM3',
    files: ['sam3_image_encoder.onnx', 'sam3_language_encoder.onnx', 'sam3_decoder.onnx'],
  },
  {
    id: 'rfdetr',
    name: 'RF-DETR',
    files: [],
  },
]

export function loadSelectedModelId() {
  const raw = localStorage.getItem(STORAGE_SELECTED)
  if (raw && MODELS.some((model) => model.id === raw)) return raw
  return MODELS[0].id
}

export function saveSelectedModelId(id: string) {
  localStorage.setItem(STORAGE_SELECTED, id)
}

export async function loadModelApi(model: AiModel): Promise<void> {
  const res = await fetch('/ai/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: model.id,
      name: model.name,
      files: model.files,
    }),
  })
  if (!res.ok) {
    throw new Error('모델 로드 API가 실패했습니다.')
  }
}
