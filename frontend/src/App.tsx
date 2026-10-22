import { useEffect, useRef, useState, type FormEvent } from 'react'
import { matchPath, useLocation, useNavigate } from 'react-router-dom'
import './App.css'
import {
  allowFolder,
  listImageHandles,
  loadFolderHandle,
  readFolderImages,
  saveFolderHandle,
  type ImageEntry,
} from './folder'
import { LabelStage } from './label'
import { MODELS, loadModelApi, loadSelectedModelId, saveSelectedModelId } from './models'
import {
  createLabelingApi,
  createProjectApi,
  formatDate,
  loadProjects,
  uploadFolderApi,
  saveProjects,
  type LabelClass,
  type LabelTool,
  type Project,
} from './projects'

const CLASS_COLORS = [
  '#ff6d2d',
  '#3b82f6',
  '#22c55e',
  '#eab308',
  '#a855f7',
  '#ef4444',
  '#14b8a6',
  '#ec4899',
  '#84cc16',
  '#6366f1',
  '#06b6d4',
  '#f43f5e',
]

function nextClassColor(items: LabelClass[]) {
  const used = new Set(items.map((item) => item.color.toLowerCase()))
  return CLASS_COLORS.find((color) => !used.has(color)) ?? CLASS_COLORS[items.length % CLASS_COLORS.length]
}

const TOOLS: { id: LabelTool; label: string; hint: string }[] = [
  { id: 'bbox', label: 'bbox', hint: '사각형' },
  { id: 'polygon', label: 'polygon', hint: '다각형' },
  { id: 'brush', label: 'brush', hint: '브러시' },
]

function ToolIcon({ id }: { id: LabelTool }) {
  if (id === 'bbox') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="6" width="16" height="12" rx="1.6" />
      </svg>
    )
  }
  if (id === 'polygon') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 3.2 20.8 8.4 18.2 20.8H5.8L3.2 8.4 12 3.2Z" />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M14.5 4.2c1.7 1.2 2.8 3.2 2.1 5.1L8.8 19.6 4.2 20.8l1.2-4.6 7.8-10.3c.4-.6.9-1.1 1.3-1.7Z" />
      <path d="M13.2 6.4 17.6 11" />
    </svg>
  )
}

function classGradient(classes: LabelClass[]) {
  const colors = classes.map((item) => item.color).filter(Boolean)
  if (colors.length === 0) return 'linear-gradient(135deg, #e4572e, #1c1917)'
  if (colors.length === 1) return `linear-gradient(135deg, ${colors[0]}, #1c1917 130%)`
  return `linear-gradient(120deg, ${colors.join(', ')})`
}

function projectPath(projectId: string, imageNumber = 1) {
  return imageNumber <= 1 ? `/projects/${projectId}` : `/projects/${projectId}/${imageNumber}`
}

function readRoute(pathname: string) {
  if (pathname === '/' || pathname === '/new') {
    return { known: true, projectId: null as string | null, imageNumber: 1, imageValid: true }
  }
  const matched = matchPath({ path: '/projects/:projectId/:image?', end: true }, pathname)
  if (!matched?.params.projectId) {
    return { known: false, projectId: null as string | null, imageNumber: 1, imageValid: true }
  }
  const raw = matched.params.image
  const imageValid = raw === undefined || /^[1-9]\d*$/.test(raw)
  return {
    known: true,
    projectId: matched.params.projectId,
    imageNumber: imageValid && raw ? Number(raw) : 1,
    imageValid,
  }
}

function App() {
  const navigate = useNavigate()
  const location = useLocation()
  const route = readRoute(location.pathname)
  const [projects, setProjects] = useState<Project[]>(() => loadProjects())
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [path, setPath] = useState('')
  const [tool, setTool] = useState<LabelTool>('bbox')
  const [opacity, setOpacity] = useState(0.4)
  const [classes, setClasses] = useState<LabelClass[]>([{ name: 'object', color: CLASS_COLORS[0] }])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [picking, setPicking] = useState(false)
  const [images, setImages] = useState<ImageEntry[]>([])
  const [imageUrl, setImageUrl] = useState('')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagesBusy, setImagesBusy] = useState(false)
  const [imagesReady, setImagesReady] = useState(false)
  const [needsFolder, setNeedsFolder] = useState(false)
  const [labelClass, setLabelClass] = useState(0)
  const [modelId, setModelId] = useState(loadSelectedModelId)
  const [loadedModelIds, setLoadedModelIds] = useState<string[]>([])
  const [modelBusy, setModelBusy] = useState(false)
  const [modelError, setModelError] = useState('')
  const [aiOn, setAiOn] = useState(false)
  const pickedFolder = useRef<FileSystemDirectoryHandle | null>(null)
  const folderHandles = useRef(new Map<string, FileSystemDirectoryHandle>())
  const imageUrlRef = useRef('')

  const modalOpen = location.pathname === '/new'
  const activeId = route.projectId
  const imageIndex = route.imageNumber - 1
  const active = projects.find((project) => project.id === activeId) ?? null
  const selectedModel = MODELS.find((model) => model.id === modelId) ?? MODELS[0]
  const modelReady = loadedModelIds.includes(selectedModel.id)

  useEffect(() => {
    setLabelClass(0)
  }, [activeId])
  const currentImage = images[imageIndex] ?? null

  function closeCreate() {
    if (!busy) navigate('/')
  }

  useEffect(() => {
    if (!route.known) {
      navigate('/', { replace: true })
      return
    }
    if (!activeId) return
    if (!projects.some((project) => project.id === activeId)) {
      navigate('/', { replace: true })
      return
    }
    const hasImageParam = location.pathname !== `/projects/${activeId}`
    if (hasImageParam && !route.imageValid) {
      navigate(projectPath(activeId), { replace: true })
      return
    }
    if (!imagesReady) return
    if (images.length === 0) {
      if (hasImageParam) navigate(projectPath(activeId), { replace: true })
      return
    }
    if (imageIndex >= images.length) navigate(projectPath(activeId), { replace: true })
  }, [activeId, imageIndex, images.length, imagesReady, location.pathname, navigate, projects, route.imageValid, route.known])

  useEffect(() => {
    if (!modalOpen) return
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape' && !busy) navigate('/')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [modalOpen, busy, navigate])

  useEffect(() => {
    if (!activeId) {
      setImages([])
      setNeedsFolder(false)
      setImagesBusy(false)
      setImagesReady(false)
      return
    }
    const projectId = activeId
    let cancelled = false
    setImages([])
    setNeedsFolder(false)
    setImagesBusy(true)
    setImagesReady(false)
    ;(async () => {
      const known = folderHandles.current.get(projectId) ?? (await loadFolderHandle(projectId))
      if (cancelled) return
      if (!known) {
        setNeedsFolder(true)
        setImagesBusy(false)
        setImagesReady(true)
        return
      }
      folderHandles.current.set(projectId, known)
      const listed = await readFolderImages(known)
      if (cancelled) return
      if (!listed) {
        setNeedsFolder(true)
        setImagesBusy(false)
        setImagesReady(true)
        return
      }
      setImages(listed)
      setNeedsFolder(false)
      setImagesBusy(false)
      setImagesReady(true)
    })().catch(() => {
      if (!cancelled) {
        setNeedsFolder(true)
        setImagesBusy(false)
        setImagesReady(true)
      }
    })
    return () => {
      cancelled = true
    }
  }, [activeId])

  useEffect(() => {
    const entry = images[imageIndex]
    if (!entry) {
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current)
      imageUrlRef.current = ''
      setImageUrl('')
      setImageFile(null)
      return
    }
    let cancelled = false
    entry.handle.getFile().then((file) => {
      if (cancelled) return
      const url = URL.createObjectURL(file)
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current)
      imageUrlRef.current = url
      setImageUrl(url)
      setImageFile(file)
    })
    return () => {
      cancelled = true
    }
  }, [images, imageIndex])

  useEffect(() => {
    return () => {
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current)
    }
  }, [])

  function persist(next: Project[]) {
    setProjects(next)
    saveProjects(next)
  }

  function resetForm() {
    setName('')
    setDescription('')
    setPath('')
    setTool('bbox')
    setOpacity(0.4)
    setClasses([{ name: 'object', color: CLASS_COLORS[0] }])
    setError('')
    pickedFolder.current = null
  }

  function openCreate() {
    resetForm()
    navigate('/new')
  }

  async function onPickFolder() {
    if (picking || busy) return
    setError('')
    let selected = ''
    try {
      const handle = await window.showDirectoryPicker()
      pickedFolder.current = handle
      selected = handle.name
      setPath(selected)
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
      setError('폴더를 선택하지 못했습니다.')
      return
    }
    setPicking(true)
    try {
      const uploaded = await uploadFolderApi(selected)
      if (uploaded) setPath(uploaded)
    } catch {
      setError('백엔드에 연결하지 못했습니다. backend를 실행한 뒤 폴더를 선택해 주세요.')
    } finally {
      setPicking(false)
    }
  }

  async function onCreate(event: FormEvent) {
    event.preventDefault()
    const trimmed = name.trim()
    const dataPath = path.trim()
    if (!trimmed) {
      setError('프로젝트 이름을 입력해 주세요.')
      return
    }
    if (!dataPath) {
      setError('데이터 폴더를 선택해 주세요.')
      return
    }
    const labelClasses = classes
      .map((item) => ({ name: item.name.trim(), color: item.color }))
      .filter((item) => item.name)
    if (labelClasses.length === 0) {
      setError('객체 종류를 하나 이상 적어 주세요.')
      return
    }
    const project: Project = {
      id: '',
      name: trimmed,
      description: description.trim(),
      path: dataPath,
      tool,
      opacity,
      classes: labelClasses,
      createdAt: new Date().toISOString(),
    }
    setBusy(true)
    setError('')
    try {
      project.id = await createProjectApi(project)
    } catch {
      setError('백엔드에 연결하지 못했습니다. backend를 실행한 뒤 다시 저장해 주세요.')
      setBusy(false)
      return
    }
    try {
      await createLabelingApi(project)
      if (pickedFolder.current) {
        folderHandles.current.set(project.id, pickedFolder.current)
        await saveFolderHandle(project.id, pickedFolder.current).catch(() => {})
      }
      persist([project, ...projects])
      navigate(projectPath(project.id), { replace: true })
      resetForm()
    } catch {
      setError('프로젝트는 만들어졌지만 라벨링 설정을 저장하지 못했습니다.')
    } finally {
      setBusy(false)
    }
  }

  function patchActive(patch: Partial<Project>) {
    if (!active) return
    persist(projects.map((item) => (item.id === active.id ? { ...item, ...patch } : item)))
  }

  async function onChangeFolder() {
    if (!active || picking || imagesBusy) return
    setPicking(true)
    try {
      const handle = await window.showDirectoryPicker()
      folderHandles.current.set(active.id, handle)
      await saveFolderHandle(active.id, handle).catch(() => {})
      patchActive({ path: handle.name })
      const listed = await readFolderImages(handle)
      setImages(listed ?? [])
      setNeedsFolder(!listed)
      if (listed && imageIndex !== 0) navigate(projectPath(active.id), { replace: true })
    } catch (error) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        setNeedsFolder(true)
      }
    } finally {
      setPicking(false)
    }
  }

  function onSelectModel(id: string) {
    setModelId(id)
    saveSelectedModelId(id)
    setModelError('')
  }

  async function onLoadModel() {
    if (modelBusy) return
    setModelBusy(true)
    setModelError('')
    try {
      await loadModelApi(selectedModel)
      setLoadedModelIds((ids) => (ids.includes(selectedModel.id) ? ids : [...ids, selectedModel.id]))
    } catch {
      setModelError('백엔드에 연결하지 못했습니다. backend를 실행한 뒤 다시 불러와 주세요.')
    } finally {
      setModelBusy(false)
    }
  }

  async function onOpenImages() {
    if (!active || imagesBusy) return
    setImagesBusy(true)
    try {
      let handle = folderHandles.current.get(active.id) ?? (await loadFolderHandle(active.id))
      if (handle && !(await allowFolder(handle))) handle = null
      if (!handle) handle = await window.showDirectoryPicker()
      folderHandles.current.set(active.id, handle)
      void saveFolderHandle(active.id, handle).catch(() => {})
      setImages(await listImageHandles(handle))
      if (imageIndex !== 0) navigate(projectPath(active.id), { replace: true })
      setNeedsFolder(false)
    } catch (error) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        setNeedsFolder(true)
      }
    } finally {
      setImagesBusy(false)
    }
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" />
          <div>
            <strong>Labeling</strong>
            <span>annotation</span>
          </div>
        </div>
        <nav>
          <button
            type="button"
            className={!active ? 'nav-item active' : 'nav-item'}
            onClick={() => navigate('/')}
          >
            <span>프로젝트</span>
            <em>{projects.length}</em>
          </button>
        </nav>
      </aside>

      <div className="main">
        <header className="topbar">
          <div>
            <p className="kicker">{active ? '프로젝트' : '작업 공간'}</p>
            <h1>{active ? active.name : 'Projects'}</h1>
          </div>
          {!active ? (
            <button type="button" className="primary" onClick={openCreate}>
              Create Project
            </button>
          ) : (
            <button type="button" className="ghost" onClick={() => navigate('/')}>
              목록으로
            </button>
          )}
        </header>

        <div className="content">
          {active ? (
            <section className="detail">
              <div className="detail-bar" style={{ background: classGradient(active.classes) }} />
              <div className="detail-body">
                {imagesBusy ? (
                  <p className="viewer-note">이미지를 읽는 중…</p>
                ) : needsFolder ? (
                  <div className="viewer-empty">
                    <p>이미지를 읽으려면 데이터 폴더를 다시 열어 주세요.</p>
                    <button type="button" className="primary" onClick={onOpenImages}>
                      폴더 열기
                    </button>
                  </div>
                ) : images.length === 0 ? (
                  <p className="viewer-note">이 폴더에서 이미지를 찾지 못했습니다.</p>
                ) : (
                  <div className="viewer">
                    {imageUrl && imageFile && currentImage ? (
                      <LabelStage
                        key={`${active.id}:${currentImage.name}`}
                        projectId={active.id}
                        imageName={currentImage.name}
                        imageUrl={imageUrl}
                        imageFile={imageFile}
                        tool={active.tool}
                        opacity={active.opacity}
                        labelClass={active.classes[labelClass] ?? active.classes[0] ?? null}
                        classes={active.classes}
                        onClassIndex={setLabelClass}
                        onTool={(next) => patchActive({ tool: next })}
                        modelId={selectedModel.id}
                        modelName={selectedModel.name}
                        modelDescription={selectedModel.description}
                        modelReady={modelReady}
                        folderPath={active.path}
                        aiOn={aiOn}
                        onAiOn={setAiOn}
                      />
                    ) : (
                      <p className="viewer-note">이미지를 읽는 중…</p>
                    )}
                    <div className="viewer-bar">
                      <button
                        type="button"
                        className="ghost"
                        onClick={() => navigate(projectPath(active.id, imageIndex))}
                        disabled={imageIndex === 0}
                      >
                        이전
                      </button>
                      <span title={currentImage?.name}>
                        {imageIndex + 1} / {images.length} {currentImage?.name}
                      </span>
                      <button
                        type="button"
                        className="ghost"
                        onClick={() => navigate(projectPath(active.id, imageIndex + 2))}
                        disabled={imageIndex >= images.length - 1}
                      >
                        다음
                      </button>
                    </div>
                  </div>
                )}
                <p className="lede">{active.description || '설명이 없습니다.'}</p>
                <dl className="meta">
                  <div>
                    <dt>경로</dt>
                    <dd className="path-edit">
                      <span className="path" title={active.path}>
                        {active.path || '없음'}
                      </span>
                      <button type="button" className="text-danger" onClick={onChangeFolder} disabled={picking}>
                        {picking ? '선택 중…' : '변경'}
                      </button>
                    </dd>
                  </div>
                  <div>
                    <dt>도구</dt>
                    <dd>
                      <select
                        aria-label="라벨 도구"
                        value={active.tool}
                        onChange={(event) => patchActive({ tool: event.target.value as LabelTool })}
                      >
                        {TOOLS.map((item) => (
                          <option key={item.id} value={item.id}>
                            {item.label}
                          </option>
                        ))}
                      </select>
                    </dd>
                  </div>
                  <div>
                    <dt>투명도</dt>
                    <dd>
                      <strong>{Math.round(active.opacity * 100)}%</strong>
                      <input
                        className="slider"
                        type="range"
                        min={0}
                        max={100}
                        style={{ ['--fill' as string]: `${Math.round(active.opacity * 100)}%` }}
                        value={Math.round(active.opacity * 100)}
                        onChange={(event) => patchActive({ opacity: Number(event.target.value) / 100 })}
                        aria-label="투명도"
                      />
                    </dd>
                  </div>
                  <div>
                    <dt>만든 날</dt>
                    <dd>{formatDate(active.createdAt)}</dd>
                  </div>
                </dl>
                <div className="class-editor">
                  {active.classes.map((item, index) => (
                    <div
                      className={index === labelClass ? 'class-row selected' : 'class-row'}
                      key={index}
                      onClick={() => setLabelClass(index)}
                    >
                      <input
                        type="color"
                        aria-label={`${index + 1}번 색`}
                        value={item.color}
                        onChange={(event) => {
                          const next = [...active.classes]
                          next[index] = { ...item, color: event.target.value }
                          patchActive({ classes: next })
                        }}
                      />
                      <input
                        type="text"
                        aria-label={`${index + 1}번 종류`}
                        value={item.name}
                        placeholder="예: door"
                        onChange={(event) => {
                          const next = [...active.classes]
                          next[index] = { ...item, name: event.target.value }
                          patchActive({ classes: next })
                        }}
                      />
                      <button
                        type="button"
                        className="text-danger"
                        onClick={() =>
                          patchActive({ classes: active.classes.filter((_, row) => row !== index) })
                        }
                      >
                        삭제
                      </button>
                    </div>
                  ))}
                  <button
                    type="button"
                    className="class-add"
                    onClick={() =>
                      patchActive({
                        classes: [...active.classes, { name: '', color: nextClassColor(active.classes) }],
                      })
                    }
                  >
                    + 종류 추가
                  </button>
                </div>
              </div>
            </section>
          ) : projects.length === 0 ? (
            <section className="empty-state">
              <div className="empty-frame" aria-hidden="true">
                <span />
              </div>
              <h2>프로젝트가 없습니다</h2>
              <p>데이터 경로를 적을 프로젝트를 먼저 만드세요.</p>
              <button type="button" className="primary" onClick={openCreate}>
                Create Project
              </button>
            </section>
          ) : (
            <ul className="cards">
              {projects.map((project) => (
                <li key={project.id}>
                  <button type="button" className="card" onClick={() => navigate(projectPath(project.id))}>
                    <div className="thumb" style={{ background: classGradient(project.classes) }}>
                      <span>{project.tool}</span>
                    </div>
                    <div className="card-body">
                      <h3>{project.name}</h3>
                      <p>{project.description || '설명 없음'}</p>
                      <footer>
                        <span className="path" title={project.path}>
                          {project.path}
                        </span>
                        <span>{formatDate(project.createdAt)}</span>
                      </footer>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <aside className="ai-panel" aria-label="모델">
        <section className="model-panel">
          <span className="model-label">모델</span>
          <p className="model-current">{selectedModel.name}</p>
          <select
            aria-label="모델 선택"
            value={selectedModel.id}
            onChange={(event) => onSelectModel(event.target.value)}
            disabled={modelBusy}
          >
            {MODELS.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
          <button type="button" className="primary" onClick={onLoadModel} disabled={modelBusy}>
            {modelBusy ? '불러오는 중…' : '불러오기'}
          </button>
          {modelReady ? <p className="model-status ready">불러옴</p> : null}
          {modelError ? (
            <p className="error" role="alert">
              {modelError}
            </p>
          ) : null}
        </section>
      </aside>

      {modalOpen ? (
        <div className="overlay" onClick={closeCreate}>
          <form
            className="modal"
            onClick={(event) => event.stopPropagation()}
            onSubmit={onCreate}
          >
            <div className="modal-head">
              <div>
                <p className="kicker">New dataset</p>
                <h2>Create Project</h2>
                <p className="lede">이름과 C++이 읽을 이미지 폴더를 선택합니다.</p>
              </div>
              <button
                type="button"
                className="close"
                aria-label="닫기"
                onClick={closeCreate}
                disabled={busy}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M6 6l12 12M18 6 6 18" />
                </svg>
              </button>
            </div>

            <div className="modal-scroll">
              <label className="field">
                <span>프로젝트 이름</span>
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="예: Door dataset"
                  autoFocus
                />
              </label>
              <label className="field">
                <span>설명</span>
                <textarea
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="무엇을 라벨할지 적어 주세요"
                  rows={3}
                />
              </label>
              <div className="field">
                <span>데이터 경로</span>
                <div className="path-pick">
                  <input
                    className="mono"
                    readOnly
                    value={path}
                    placeholder="폴더를 선택해 주세요"
                    onClick={onPickFolder}
                  />
                  <button
                    type="button"
                    className="browse"
                    onClick={onPickFolder}
                    disabled={picking || busy}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M3.5 7.5h6l2 2h9v9.5a1.5 1.5 0 0 1-1.5 1.5h-14A1.5 1.5 0 0 1 3.5 19V7.5Z" />
                      <path d="M3.5 7.5V6A1.5 1.5 0 0 1 5 4.5h4l2 2" />
                    </svg>
                    {picking ? '선택 중…' : '폴더 선택'}
                  </button>
                </div>
              </div>

              <section className="panel">
                <div className="panel-label">라벨 도구</div>
                <div className="tools" role="radiogroup" aria-label="라벨 도구">
                  {TOOLS.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      role="radio"
                      aria-checked={tool === item.id}
                      className={tool === item.id ? 'tool active' : 'tool'}
                      onClick={() => setTool(item.id)}
                    >
                      <ToolIcon id={item.id} />
                      <strong>{item.label}</strong>
                      <small>{item.hint}</small>
                    </button>
                  ))}
                </div>
              </section>

              <section className="panel">
                <div className="opacity-head">
                  <span>투명도</span>
                  <div className="opacity-readout">
                    <i
                      style={{
                        background: classes[0]?.color ?? '#ff6d2d',
                        opacity,
                      }}
                    />
                    <strong>{Math.round(opacity * 100)}%</strong>
                  </div>
                </div>
                <input
                  className="slider"
                  type="range"
                  min={0}
                  max={100}
                  style={{ ['--fill' as string]: `${Math.round(opacity * 100)}%` }}
                  value={Math.round(opacity * 100)}
                  onChange={(event) => setOpacity(Number(event.target.value) / 100)}
                  aria-label="투명도"
                />
              </section>

              <section className="panel">
                <div className="panel-label">객체 종류</div>
                {classes.map((item, index) => (
                  <div className="class-row" key={index}>
                    <input
                      type="color"
                      aria-label={`${index + 1}번 색`}
                      value={item.color}
                      onChange={(event) => {
                        const next = [...classes]
                        next[index] = { ...item, color: event.target.value }
                        setClasses(next)
                      }}
                    />
                    <input
                      type="text"
                      aria-label={`${index + 1}번 종류`}
                      value={item.name}
                      placeholder="예: door"
                      onChange={(event) => {
                        const next = [...classes]
                        next[index] = { ...item, name: event.target.value }
                        setClasses(next)
                      }}
                    />
                    <button
                      type="button"
                      className="text-danger"
                      onClick={() => setClasses(classes.filter((_, row) => row !== index))}
                    >
                      삭제
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  className="class-add"
                  onClick={() => setClasses([...classes, { name: '', color: nextClassColor(classes) }])}
                >
                  + 종류 추가
                </button>
              </section>

              {error ? (
                <p className="error" role="alert">
                  {error}
                </p>
              ) : null}
            </div>

            <div className="modal-footer">
              <button type="button" className="ghost" onClick={closeCreate} disabled={busy}>
                Cancel
              </button>
              <button type="submit" className="primary" disabled={busy}>
                {busy ? '저장 중…' : 'Save'}
              </button>
            </div>
          </form>
        </div>
      ) : null}
    </div>
  )
}

export default App
