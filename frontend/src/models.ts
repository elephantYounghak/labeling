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
    description: '클릭한 위치의 물체 하나를 찾아 그립니다.',
    files: ['vision_encoder_q4f16.onnx', 'prompt_encoder_mask_decoder_int8.onnx'],
  },
  {
    id: 'sam3_text',
    name: 'SAM3 Text',
    description: '이미지를 클릭하면 이미지 전체에서 물체를 모두 찾아 그립니다.',
    files: ['sam3_image_encoder.onnx', 'sam3_language_encoder.onnx', 'sam3_decoder.onnx'],
  },
  {
    id: 'rfdetr',
    name: 'RF-DETR',
    description: '테스트용 모델입니다. 아직 검출 결과가 없습니다.',
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
