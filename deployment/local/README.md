# Foldy Local Deployment

Run Foldy locally with a single command - no git clone required!

---

# 🚀 For Developers: Release New Version

**To release a new version to DockerHub RIGHT NOW:**

```bash
# 1. Login to DockerHub (one-time setup)
docker login

# 2. Build and push new version (from repo root)
cd deployment/local
./build_and_deploy_containers.sh v2.1.0

# 3. Test the new version
FOLDY_STORAGE_DIRECTORY=/tmp/foldy-test FOLDY_VERSION=v2.1.0 \
  docker-compose up -d
```

**See [Developer Guide](#developer-guide) below for full details.**

---

# 👤 For Users: Run Foldy Locally

## Quick Start

```bash
# Create data directory and run Foldy
FOLDY_STORAGE_DIRECTORY=$HOME/foldy-data \
  docker-compose -f <(curl -s https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml) up -d
```

That's it! Foldy will be available at **http://localhost:3000**

## Requirements

- Docker and Docker Compose installed
- At least 8GB RAM recommended
- ~50GB free disk space for models and data
- Internet connection for initial setup

## Configuration

### Required Environment Variables

- `FOLDY_STORAGE_DIRECTORY` - Path where Foldy will store all persistent data

### Optional Environment Variables

- `FOLDY_VERSION=latest` - Docker image version to use
- `FOLDY_DOCKERHUB_USER=jbrlbl` - DockerHub username for images

## Examples

```bash
# Use specific version
FOLDY_STORAGE_DIRECTORY=~/foldystorage FOLDY_VERSION=v2.0.0 \
  docker-compose -f <(curl -s https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml) up -d

# Set admin email
FOLDY_STORAGE_DIRECTORY=$HOME/foldy-data \
  docker-compose -f <(curl -s https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml) up -d

# Download and run locally
curl -O https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml
FOLDY_STORAGE_DIRECTORY=/data/foldy docker-compose up -d
```

## Data Storage

Your `FOLDY_STORAGE_DIRECTORY` will contain:

```
foldy-data/
├── postgres_data/    # PostgreSQL database files
├── blob_storage/     # Protein structures, results, uploads
└── boltz_cache/      # Cached Boltz model files (~40GB when populated)
```

## Management Commands

```bash
# Stop Foldy
docker-compose down

# View logs
docker-compose logs -f

# Update to latest version
docker-compose pull && docker-compose up -d

# Restart specific service
docker-compose restart backend

# Database shell
docker-compose exec db psql -U user -d postgres

# Backend shell
docker-compose exec backend bash
```

## Troubleshooting

### Services fail to start
1. Check Docker is running: `docker info`
2. Ensure `FOLDY_STORAGE_DIRECTORY` exists and is writable
3. Try full restart: `docker-compose down && docker-compose up -d`

### "No space left on device"
- Foldy needs ~50GB for model downloads
- Clean up: `docker system prune -a`
- Check disk space: `df -h`

### Database connection issues
- Wait 30-60 seconds for PostgreSQL to initialize
- Check logs: `docker-compose logs db`
- Verify data directory permissions

### Slow performance
- Ensure adequate RAM (8GB minimum, 16GB recommended)
- Check CPU usage during model operations
- Consider SSD storage for better I/O

---

# 🛠 Developer Guide

## Setup

### 1. DockerHub Access
```bash
# Login to DockerHub (required for pushing images)
docker login

# Verify access to jbrlbl organization
docker search jbrlbl
```

### 2. Repository Setup
```bash
# Clone repository
git clone https://github.com/JBEI/foldy.git
cd foldy

# Ensure you're on the right branch
git checkout main
git pull origin main
```

## Release Process

### 1. Test Development Build
```bash
# Test with development compose first
docker-compose -f deployment/development/docker-compose.yml up -d
# ... verify everything works ...
docker-compose -f deployment/development/docker-compose.yml down
```

### 2. Build and Push Release
```bash
cd deployment/local

# Build and push to DockerHub (replace with your version)
./build_and_deploy_containers.sh v2.1.0

# Or use environment variables for custom settings
DOCKERHUB_USER=myusername BACKEND_URL=https://myfoldy.com \
  ./build_and_deploy_containers.sh v2.1.0
```

### 3. Test Production Images
```bash
# Test the newly pushed images
FOLDY_STORAGE_DIRECTORY=/tmp/foldy-test FOLDY_VERSION=v2.1.0 \
  docker-compose up -d

# Verify services are healthy
docker-compose ps
docker-compose logs

# Clean up test
docker-compose down
rm -rf /tmp/foldy-test
```

### 4. Update Version References
```bash
# Update any documentation or scripts that reference the version
# Consider updating the default FOLDY_VERSION in docker-compose.yml if needed
```

## Build Script Options

The `build_and_deploy_containers.sh` script accepts these environment variables:

- `DOCKERHUB_USER` - DockerHub username (default: jbrlbl)
- `DOCKERHUB_REPO_PREFIX` - Image name prefix (default: foldy)
- `BACKEND_URL` - Backend URL for frontend build (default: http://localhost:8080)
- `INSTITUTION` - Institution name for frontend (default: "Foldy Local")

## Docker Images Built

The script builds and pushes these images:

- `jbrlbl/foldy-frontend:VERSION`
- `jbrlbl/foldy-backend:VERSION`
- `jbrlbl/foldy-worker-esm:VERSION`
- `jbrlbl/foldy-worker-boltz:VERSION`

## Testing Checklist

Before releasing, verify:

- [ ] All services start successfully
- [ ] Frontend accessible at http://localhost:3000
- [ ] Backend API responds at http://localhost:8080
- [ ] Database migrations run successfully
- [ ] Workers can process jobs (test with sample protein)
- [ ] File uploads work correctly
- [ ] Data persists after restart

## Version Tagging Strategy

- Use semantic versioning: `v1.2.3`
- `latest` tag is automatically updated with each push
- Keep `latest` stable - use feature branches for experimental builds
- Document breaking changes in release notes

## Troubleshooting Builds

### Build fails with "no space left on device"
```bash
# Clean up Docker
docker system prune -a
docker volume prune
```

### Push fails with authentication error
```bash
# Re-login to DockerHub
docker logout
docker login
```

### Image too large
```bash
# Check image sizes
docker images | grep foldy

# Optimize Dockerfiles if needed
# Consider multi-stage builds to reduce final image size
```
