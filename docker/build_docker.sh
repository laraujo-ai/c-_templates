set -e

DOCKERFILE="docker/DockerFile.dev_5_1_3"
IMAGE_NAME="cpp-pipeline"
IMAGE_TAG="latest"
BASE_IMAGE="nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel"
PARALLEL_JOBS="4"
ONNXRUNTIME_TAG="v1.16.0"
NO_CACHE=""

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build the C++ video-analytics pipeline Docker container

OPTIONS:
    -h, --help              Show this help message
    -t, --tag TAG           Docker image tag (default: latest)
    -j, --jobs NUM          Number of parallel build jobs (default: 4)
    -b, --base-image IMG    Base Docker image
    --onnx-tag TAG          ONNX Runtime version tag (default: v1.16.0)
    --no-cache              Build without using cache
    --x86                   Use x86_64 base image instead of ARM64

EXAMPLES:
    $0                      # Basic build for Jetson (ARM64)
    $0 -t v1.0              # Build with custom tag
    $0 --x86                # Build for x86_64 with GPU
    $0 -j 2 --no-cache      # Clean build with fewer jobs

EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -t|--tag) IMAGE_TAG="$2"; shift 2 ;;
        -j|--jobs) PARALLEL_JOBS="$2"; shift 2 ;;
        -b|--base-image) BASE_IMAGE="$2"; shift 2 ;;
        --onnx-tag) ONNXRUNTIME_TAG="$2"; shift 2 ;;
        --no-cache) NO_CACHE="--no-cache"; shift ;;
        --x86) BASE_IMAGE="nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04"; shift ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

echo "Building ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Base: ${BASE_IMAGE}"
echo "  Jobs: ${PARALLEL_JOBS}"
echo "  ONNX Runtime: ${ONNXRUNTIME_TAG}"
echo ""

if [ ! -f "${DOCKERFILE}" ]; then
    echo "Error: Dockerfile not found at ${DOCKERFILE}"
    exit 1
fi

docker build \
    ${NO_CACHE} \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg PARALLEL_JOBS="${PARALLEL_JOBS}" \
    --build-arg ONNXRUNTIME_TAG="${ONNXRUNTIME_TAG}" \
    .
