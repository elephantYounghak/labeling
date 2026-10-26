# xmake

백엔드(`backend/xmake.lua`)는 xmake로 빌드한다. `backend`에서 `xmake`는 패키지를 맞춘 뒤 컴파일하고, `xmake run`은 그 바이너리를 실행한다. 설치 확인이 뜨면 `xmake -y`로 그대로 받는다. 루트의 `run.ps1`도 같은 순서로 `xmake` 다음 `xmake run`을 호출한다.

패키지 파일은 `%LOCALAPPDATA%\.xmake\packages`에 있다. `backend/.xmake`는 이 프로젝트의 설정 캐시라서, 이 폴더를 지워도 이미 설치된 패키지는 남고 다음 실행 때 캐시만 다시 생긴다.

`system`은 패키지를 어디서 가져올지 정한다.

- 값을 비우면 패키지 저장소를 먼저 보고, 없을 때 컴퓨터에 설치된 라이브러리를 찾는다.
- `false`는 컴퓨터에 설치된 것을 확인하지 않는다. 저장소에 있으면 그걸 쓰고, 없으면 xmake가 설치한다.
- `true`는 설치하지 않고, 컴퓨터에 있는 것만 찾는다. 없으면 빌드가 멈춘다.

이 프로젝트는 cuDNN만 `system = true`다. `add_requireconfs("**|cudnn", {system = false})`가 나머지 전부(의존성 포함)를 `false`로 덮는다. cuDNN 9.24는 `C:/Program Files/NVIDIA/CUDNN/v9.24`의 `include/12.8`, `lib/12.8/x64`, `bin/12.8/x64`를 읽는다.

CUDA는 12.8.1 하나로 고정한다. `add_requires("cuda 12.8.1")`가 직접 의존이고, `add_requireconfs("**cuda")`가 OpenCV처럼 다른 패키지가 끌어오는 CUDA까지 같은 버전과 `cudart`, `cublas`, `cufft`, `nvrtc`로 맞춘다. OpenCV 패키지 스크립트는 CUDA를 `system = true`로 선언하므로, 위의 `system = false` override가 그 선언을 덮고 xmake가 CUDA를 패키지 저장소에 설치하게 한다.

그 외 요구 버전은 Drogon(trantor v1.5.28), SOCI(ODBC), ONNX Runtime 1.22.1(GPU), OpenCV(shared, CUDA, ffmpeg/opengl/vtk 끔)다.

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend enabling type-aware lint rules by installing `oxlint-tsgolint` and editing `.oxlintrc.json`:

```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["react", "typescript", "oxc"],
  "options": {
    "typeAware": true
  },
  "rules": {
    "react/rules-of-hooks": "error",
    "react/only-export-components": ["warn", { "allowConstantExport": true }]
  }
}
```

See the [Oxlint rules documentation](https://oxc.rs/docs/guide/usage/linter/rules) for the full list of rules and categories.
