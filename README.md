# C++ Video Analytics Pipeline Template

A starting point for building GPU-accelerated video analytics applications on NVIDIA Jetson (JetPack 5.x / 6.x) or x86 with CUDA.

The template provides the full build stack out of the box: GStreamer ingestion, ONNX Runtime inference (TensorRT + CUDA execution providers), object detection, SORT tracking, and an RTSP relay via MediaMTX — with nothing project-specific baked in.

---

## What's included

| Layer | Component | Notes |
|---|---|---|
| Ingestion | `stream_handler` | GStreamer RTSP + file source, per-frame queue (`FrameContainer`) |
| Inference | `common` (base_model / onnx_session) | Template base class for any ONNX model |
| Detection | `object_detection` (YOLOX) | Swap for your own detector by subclassing `IBaseModel` |
| Tracking | `tracker` (SORT) | Eigen3-based Kalman filter tracker |
| Logging | spdlog (vendored) | `LOG_INFO / LOG_ERROR / …` macros, zero extra deps |
| Testing | Catch2 (vendored) | Unit tests per component |
| RTSP relay | MediaMTX | Useful for re-streaming sources during development |

---

## Project layout

```
src/
  common/             # IBaseModel<In,Out>, ONNXSessionBuilder, logger, metrics
  components/
    stream_handler/   # GStreamerRTSPHandler / GStreamerFileHandler
    object_detection/ # YOLOXDetector — replace or extend for your model
    tracker/          # SortTracker
  lib/
    spdlog/           # vendored (header-only)
    catch2/           # vendored (amalgamated)
docker/
  DockerFile.dev_5_1_3   # JetPack 5.x (TensorRT r8.5, ONNX RT 1.16)
  DockerFile.dev_6_0     # JetPack 6.x (CUDA 12.2, ONNX RT 1.17)
  build_docker.sh        # build helper
deploy_5_1_3.yml         # docker-compose for JetPack 5.x
deploy_6_0.yml           # docker-compose for JetPack 6.x
mediamtx.yml             # MediaMTX configuration
```

---

## Getting started

### 1. Build the Docker image

```bash
# JetPack 5.x (default)
bash docker/build_docker.sh

# JetPack 6.x
bash docker/build_docker.sh \
    -b nvcr.io/nvidia/l4t-jetpack:r36.3.0 \
    --onnx-tag v1.17.1

# x86 with CUDA
bash docker/build_docker.sh --x86

# Custom tag / parallel jobs
bash docker/build_docker.sh -t v1.0 -j 8
```

The image is tagged `cpp-pipeline:<tag>`. The Dockerfile builds the template source automatically — the binary lands at `/workspace/build/`.

### 2. Add your application code

1. **Model** — drop your `.onnx` file into the project and mount it at runtime (see step 3).
2. **Detector** — subclass `IBaseModel<cv::Mat, std::vector<Detection>>` in `src/components/object_detection/`, or add a new component under `src/components/`.
3. **Entry point** — write your `main.cpp` under `src/lib/` and register it with `add_executable` in the root `CMakeLists.txt`.

The CMake build receives the ONNX Runtime paths automatically from the Dockerfile:
```bash
cmake ../src \
    -DONNXRUNTIME_LIB=/opt/onnxruntime/lib/libonnxruntime.so \
    -DONNXRUNTIME_INCLUDE_DIR=/opt/onnxruntime/include
```

### 3. Run with docker-compose

```bash
# Start mediamtx + your app container
docker compose -f deploy_5_1_3.yml up
```

- MediaMTX is available on `:8554` (RTSP) and `:9997` (REST API).
- The `cpp-app` container starts a `bash` shell by default. Once you have a binary, set the `command:` field in the compose file to run it.

Mount a model and config at runtime by adding volumes to the `cpp-app` service:
```yaml
volumes:
  - ./models:/workspace/models:ro
  - ./config.json:/workspace/config.json:ro
```

---

## Adding a new component

```
src/components/my_component/
  include/my_component.hpp
  src/my_component.cpp
  tests/
    CMakeLists.txt
    test_my_component.cpp
  CMakeLists.txt
```

Follow the same pattern as existing components (`add_library` + `target_link_libraries`), then add `add_subdirectory(my_component)` to [src/components/CMakeLists.txt](src/components/CMakeLists.txt).

---

## Namespace

All template code lives under `namespace project_x`. Rename it globally when starting a new project.
