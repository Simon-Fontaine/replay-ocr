# Variables
IMAGE_NAME = simonfontaine/replay-ocr-app
TAG = latest
DOCKER_FULL_NAME = $(IMAGE_NAME):$(TAG)

# Phony targets
.PHONY: help build push deploy clean

# Help
help:
	@echo "Available commands:"
	@echo "  make build   - Build Docker image"
	@echo "  make push    - Push image to Docker Hub"
	@echo "  make deploy  - Build, push and deploy to Railway"
	@echo "  make clean   - Remove local Docker image"

# Build
build:
	@echo "Building Docker image $(DOCKER_FULL_NAME)..."
	docker build -t $(DOCKER_FULL_NAME) .
#	docker build --no-cache -t $(DOCKER_FULL_NAME) .

# Push
push:
	@echo "Pushing $(DOCKER_FULL_NAME) to Docker Hub..."
	docker push $(DOCKER_FULL_NAME)

# Deploy
deploy: build push
	@echo "Deploying to Railway..."
	railway run --railway-env .env railway.toml

# Clean
clean:
	@echo "Removing local Docker image $(DOCKER_FULL_NAME)..."
	docker rmi $(DOCKER_FULL_NAME) || true
	@echo "Clean complete!"