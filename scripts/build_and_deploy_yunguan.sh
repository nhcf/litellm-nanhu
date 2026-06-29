#!/bin/bash
# 运管认证集成版本构建和部署脚本
# 使用方法: ./scripts/build_and_deploy_yunguan.sh [镜像标签]

set -e

# 配置变量
REGISTRY="10.200.93.79:15080"
IMAGE_NAME="gyjc/litellm"
NAMESPACE="litellm-system"

# 获取当前 git commit short hash 作为版本号
GIT_HASH=$(git rev-parse --short HEAD)
VERSION="${1:-v1.85.0-${GIT_HASH}-yunguan}"

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

echo "========================================"
echo "构建运管认证集成版本"
echo "镜像: ${FULL_IMAGE}"
echo "========================================"

# 1. 构建 Docker 镜像
echo "[1/4] 构建 Docker 镜像..."
docker build \
    --build-arg HTTP_PROXY="${HTTP_PROXY:-}" \
    --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
    --build-arg NO_PROXY="${NO_PROXY:-}" \
    -t "${FULL_IMAGE}" \
    -f Dockerfile \
    .

# 2. 推送镜像到仓库
echo "[2/4] 推送镜像到仓库..."
docker push "${FULL_IMAGE}"

# 3. 更新 K8s 部署
echo "[3/4] 更新 K8s 部署..."
kubectl set image deployment/litellm \
    litellm="${FULL_IMAGE}" \
    -n "${NAMESPACE}"

# 4. 等待部署就绪
echo "[4/4] 等待部署就绪..."
kubectl rollout status deployment/litellm -n "${NAMESPACE}" --timeout=120s

echo "========================================"
echo "部署完成！"
echo "镜像版本: ${VERSION}"
echo "验证命令: curl http://10.200.44.107:37400/health"
echo "========================================"