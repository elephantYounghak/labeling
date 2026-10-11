export type ListedItem = {
  path: string
  kind: 'file' | 'directory'
}

const MAX_ITEMS = 500

export async function listDirectory(
  dir: FileSystemDirectoryHandle,
  prefix = '',
): Promise<ListedItem[]> {
  const items: ListedItem[] = []

  for await (const [name, handle] of dir.entries()) {
    const path = prefix ? `${prefix}/${name}` : name
    if (handle.kind === 'directory') {
      items.push({ path, kind: 'directory' })
      items.push(...(await listDirectory(handle, path)))
    } else {
      items.push({ path, kind: 'file' })
    }
    if (items.length >= MAX_ITEMS) {
      break
    }
  }

  return items
}

export function listFromInputFiles(files: FileList): {
  folderName: string
  items: ListedItem[]
} {
  const list = Array.from(files)
  const first = list[0]?.webkitRelativePath ?? ''
  const folderName = first.split('/')[0] || '선택한 폴더'
  const items = list.slice(0, MAX_ITEMS).map((file) => ({
    path: file.webkitRelativePath || file.name,
    kind: 'file' as const,
  }))
  return { folderName, items }
}

export function canPickDirectory(): boolean {
  return typeof window.showDirectoryPicker === 'function'
}

const IMAGE_EXT = new Set(['png', 'jpg', 'jpeg', 'bmp', 'webp', 'gif', 'tif', 'tiff'])

export type ImageEntry = {
  name: string
  handle: FileSystemFileHandle
}

export async function listImageHandles(dir: FileSystemDirectoryHandle): Promise<ImageEntry[]> {
  const images: ImageEntry[] = []
  await collectImages(dir, '', images)
  images.sort((a, b) => a.name.localeCompare(b.name, 'ko'))
  return images
}

async function collectImages(
  dir: FileSystemDirectoryHandle,
  prefix: string,
  images: ImageEntry[],
) {
  for await (const [name, handle] of dir.entries()) {
    if (images.length >= 2000 || name.startsWith('.')) continue
    const path = prefix ? `${prefix}/${name}` : name
    if (handle.kind === 'directory') {
      await collectImages(handle, path, images)
      continue
    }
    const ext = name.slice(name.lastIndexOf('.') + 1).toLowerCase()
    if (!IMAGE_EXT.has(ext)) continue
    images.push({ name: path, handle })
  }
}

const HANDLE_DB = 'labeling'
const HANDLE_STORE = 'folders'

function openHandleDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(HANDLE_DB, 1)
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(HANDLE_STORE)) {
        request.result.createObjectStore(HANDLE_STORE)
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

export async function saveFolderHandle(projectId: string, handle: FileSystemDirectoryHandle) {
  const db = await openHandleDb()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(HANDLE_STORE, 'readwrite')
    tx.objectStore(HANDLE_STORE).put(handle, projectId)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
  db.close()
}

export async function loadFolderHandle(projectId: string): Promise<FileSystemDirectoryHandle | null> {
  const db = await openHandleDb()
  const handle = await new Promise<FileSystemDirectoryHandle | undefined>((resolve, reject) => {
    const request = db.transaction(HANDLE_STORE, 'readonly').objectStore(HANDLE_STORE).get(projectId)
    request.onsuccess = () => resolve(request.result as FileSystemDirectoryHandle | undefined)
    request.onerror = () => reject(request.error)
  })
  db.close()
  return handle ?? null
}

function accessDenied(error: unknown) {
  return error instanceof DOMException && (error.name === 'NotAllowedError' || error.name === 'SecurityError')
}

async function permissionGranted(handle: FileSystemDirectoryHandle, ask: boolean) {
  for (const mode of ['read', 'readwrite'] as const) {
    try {
      const state = ask
        ? await handle.requestPermission({ mode })
        : await handle.queryPermission({ mode })
      if (state === 'granted') return true
    } catch {
      continue
    }
  }
  return false
}

export async function readFolderImages(handle: FileSystemDirectoryHandle): Promise<ImageEntry[] | null> {
  try {
    return await listImageHandles(handle)
  } catch (error) {
    if (!accessDenied(error)) throw error
  }
  if (!(await permissionGranted(handle, false)) && !(await permissionGranted(handle, true))) return null
  try {
    return await listImageHandles(handle)
  } catch (error) {
    if (!accessDenied(error)) throw error
    return null
  }
}

export async function allowFolder(handle: FileSystemDirectoryHandle): Promise<boolean> {
  return permissionGranted(handle, true)
}
