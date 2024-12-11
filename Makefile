# Variables
APP_NAME = app
DOCKER_COMPOSE = docker-compose

# Colors
BLUE := \033[1;34m
GREEN := \033[1;32m
YELLOW := \033[1;33m
RED := \033[1;31m
RESET := \033[0m

# Default target (help)
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "$(GREEN)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"; printf "  $(BLUE)%-15s$(RESET) %s\n", "Target", "Description"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# Docker commands
.PHONY: up
up: ## Start the containers in detached mode
	@echo "$(YELLOW)Starting containers...$(RESET)"
	$(DOCKER_COMPOSE) up -d

.PHONY: down
down: ## Stop and remove containers
	@echo "$(YELLOW)Stopping and removing containers...$(RESET)"
	$(DOCKER_COMPOSE) down

.PHONY: build
build: ## Build the Docker images
	@echo "$(YELLOW)Building Docker images...$(RESET)"
	$(DOCKER_COMPOSE) build

.PHONY: rebuild
rebuild: down build up ## Rebuild and restart the containers
	@echo "$(YELLOW)Rebuilding containers...$(RESET)"

.PHONY: restart
restart: ## Restart the application container
	@echo "$(YELLOW)Restarting $(APP_NAME) service...$(RESET)"
	$(DOCKER_COMPOSE) restart $(APP_NAME)

.PHONY: logs
logs: ## Tail the logs of all services
	@echo "$(YELLOW)Tailing logs for all services...$(RESET)"
	$(DOCKER_COMPOSE) logs -f

.PHONY: logs-app
logs-app: ## Tail the logs of the app service
	@echo "$(YELLOW)Tailing logs for the $(APP_NAME) service...$(RESET)"
	$(DOCKER_COMPOSE) logs -f $(APP_NAME)

.PHONY: exec
exec: ## Access the app container shell
	@echo "$(YELLOW)Accessing the $(APP_NAME) container shell...$(RESET)"
	$(DOCKER_COMPOSE) exec $(APP_NAME) /bin/bash

# Cleanup commands
.PHONY: clean
clean: ## Stop and clean containers, networks, and volumes
	@echo "$(RED)Cleaning up containers, networks, and volumes...$(RESET)"
	$(DOCKER_COMPOSE) down -v

.PHONY: prune
prune: ## Remove unused Docker resources
	@echo "$(RED)Pruning unused Docker resources...$(RESET)"
	docker system prune -a -f --volumes

# Monitoring and system info
.PHONY: health
health: ## Check the health of the app container
	@echo "$(BLUE)Checking health of the application...$(RESET)"
	curl -s http://localhost:8080/health | jq

.PHONY: stats
stats: ## Show resource usage stats for containers
	@echo "$(BLUE)Displaying container resource usage stats...$(RESET)"
	docker stats

.PHONY: ps
ps: ## List running containers
	@echo "$(BLUE)Listing running containers...$(RESET)"
	$(DOCKER_COMPOSE) ps
