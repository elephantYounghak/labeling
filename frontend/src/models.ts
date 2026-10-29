export type AiModel = {
  id: string
  name: string
  description: string
  files: string[]
}

const STORAGE_SELECTED = 'labeling.model'

export const MODELS: AiModel[] = [
  {
    id: 'sam3_point',
    name: 'SAM3 Point',
    description: 'Alt+클릭한 위치의 물체 하나를 찾아 그립니다.',
    files: ['vision_encoder_q4f16.onnx', 'prompt_encoder_mask_decoder_int8.onnx'],
  },
  {
    id: 'sam3_text',
    name: 'SAM3 Text',
    description: 'Alt+클릭하면 이미지 전체에서 물체를 모두 찾아 그립니다.',
    files: ['vision_encoder_int4.onnx', 'text_encoder_int4.onnx', 'decoder_int4.onnx'],
  },
  {
    id: 'rfdetr',
    name: 'RF-DETR',
    description: '라벨한 이미지로 학습한 뒤, 라벨 없는 이미지에 박스를 그립니다.',
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
    const body = await res.json().catch(() => null)
    const detail = body && typeof body.error === 'string' ? body.error : ''
    throw new Error(detail || '모델 로드 API가 실패했습니다.')
  }
}
