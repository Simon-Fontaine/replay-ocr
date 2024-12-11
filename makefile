# Configuration
IMAGE_NAME = simonfontaine/replay-ocr-app
TAG = latest
DOCKER_FULL_NAME = $(IMAGE_NAME):$(TAG)

# Declare phony targets (targets that don't create files)
.PHONY: help build push deploy clean

# Default target when just running 'make'
.DEFAULT_GOAL := help

# Help command that lists all available commands
help:
	@echo "Available commands:"
	@echo "make build   - Build Docker image"
	@echo "make push    - Push image to Docker Hub"
	@echo "make deploy  - Build, push and deploy to Railway"
	@echo "make clean   - Remove local Docker image"

# Build the Docker image
build:
	@echo "Building Docker image $(DOCKER_FULL_NAME)..."
	docker build --no-cache -t $(DOCKER_FULL_NAME) .
	@echo "Build complete!"

# Push the image to Docker Hub
push:
	@echo "Pushing $(DOCKER_FULL_NAME) to Docker Hub..."
	docker push $(DOCKER_FULL_NAME)
	@echo "Push complete!"

# Deploy to Railway (includes build and push)
deploy: build push
	@echo "Deploying to Railway..."
	railway up
	@echo "Deployment complete!"

# Clean up local Docker images
clean:
	@echo "Removing local Docker image $(DOCKER_FULL_NAME)..."
	docker rmi $(DOCKER_FULL_NAME) || true
	@echo "Clean complete!"