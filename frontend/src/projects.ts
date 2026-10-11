export type LabelTool = 'bbox' | 'polygon' | 'brush'

export type LabelClass = {
  name: string
  color: string
}

export type Project = {
  id: string
  name: string
  description: string
  path: string
  tool: LabelTool
  opacity: number
  classes: LabelClass[]
  createdAt: string
}

type StoredProject = Partial<Project> & {
  folderName?: string
}

const STORAGE_KEY = 'labeling.projects'

function readTool(value: unknown): LabelTool {
  if (value === 'bbox' || value === 'polygon' || value === 'brush') {
    return value
  }
  return 'bbox'
}

function readOpacity(value: unknown): number {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 0.4
  }
  return Math.min(1, Math.max(0, value))
}

function readClasses(value: unknown): LabelClass[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.flatMap((item) => {
    if (!item || typeof item !== 'object') {
      return []
    }
    const row = item as { name?: unknown; color?: unknown }
    const name = typeof row.name === 'string' ? row.name : ''
    const color = typeof row.color === 'string' ? row.color : '#ff6d2d'
    return name ? [{ name, color }] : []
  })
}

export function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return []
    }
    const parsed = JSON.parse(raw) as StoredProject[]
    if (!Array.isArray(parsed)) {
      return []
    }
    return parsed.map((item) => ({
      id: item.id == null ? '' : String(item.id),
      name: item.name ?? '',
      description: item.description ?? '',
      path: item.path ?? item.folderName ?? '',
      tool: readTool(item.tool),
      opacity: readOpacity(item.opacity),
      classes: readClasses(item.classes),
      createdAt: item.createdAt ?? new Date().toISOString(),
    }))
  } catch {
    return []
  }
}

export function saveProjects(projects: Project[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(projects))
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

export async function uploadFolderApi(currentPath: string): Promise<string> {
  const res = await fetch('/api/upload-folder', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: currentPath }),
  })
  if (!res.ok) {
    throw new Error('폴더 선택 API가 실패했습니다.')
  }
  const data = (await res.json()) as { path?: unknown }
  return typeof data.path === 'string' ? data.path : ''
}

export async function createProjectApi(project: Project): Promise<string> {
  const res = await fetch('/api/projects', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: project.name,
      description: project.description,
      path: project.path,
      createdAt: project.createdAt,
    }),
  })
  if (!res.ok) {
    throw new Error('프로젝트 저장 API가 실패했습니다.')
  }
  const data = (await res.json()) as { id?: unknown }
  if (typeof data.id !== 'string' && typeof data.id !== 'number') {
    throw new Error('프로젝트 번호를 받지 못했습니다.')
  }
  return String(data.id)
}

export async function createLabelingApi(project: Project): Promise<void> {
  const res = await fetch('/api/labeling', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      projectId: project.id,
      tool: project.tool,
      opacity: project.opacity,
      classes: project.classes,
    }),
  })
  if (!res.ok) {
    throw new Error('라벨링 저장 API가 실패했습니다.')
  }
}
