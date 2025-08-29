#!/bin/bash
# Cloud Deployment Script for Trading Bot
# Supports multiple cloud platforms and deployment strategies

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PLATFORM=""
ENVIRONMENT="production"
MEMORY="2G"
CPU="1.0"
AUTO_SCALING="false"

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    cat << EOF
🚀 Trading Bot Cloud Deployment Script

Usage: $0 --platform PLATFORM [OPTIONS]

Required:
  --platform PLATFORM     Target platform (railway|render|digitalocean|aws|gcp)

Options:
  --environment ENV        Environment (development|staging|production) [default: production]
  --memory MEMORY          Memory allocation (e.g., 2G, 1024M) [default: 2G]
  --cpu CPU                CPU allocation (e.g., 1.0, 0.5) [default: 1.0]
  --auto-scaling           Enable auto-scaling (where supported)
  --help                   Show this help message

Supported Platforms:
  railway       - Deploy to Railway.app (easiest)
  render        - Deploy to Render.com (free tier available)
  digitalocean  - Deploy to DigitalOcean Droplet
  aws           - Deploy to AWS ECS Fargate
  gcp           - Deploy to Google Cloud Run

Examples:
  $0 --platform railway
  $0 --platform digitalocean --memory 4G --cpu 2.0
  $0 --platform aws --environment staging --auto-scaling

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if .env file exists
    if [[ ! -f ".env" ]]; then
        log_error ".env file not found!"
        log_info "Please copy env_template.txt to .env and fill in your API keys"
        exit 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed!"
        log_info "Please install Docker from https://docker.com"
        exit 1
    fi
    
    # Check if required files exist
    required_files=("Dockerfile" "docker-compose.yml" "main.py" "config/config.yaml")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file $file not found!"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed!"
}

validate_env_variables() {
    log_info "Validating environment variables..."
    
    # Check for critical API keys
    source .env
    
    critical_vars=("ALPHA_VANTAGE_API_KEY" "OPENAI_API_KEY")
    missing_vars=()
    
    for var in "${critical_vars[@]}"; do
        if [[ -z "${!var}" || "${!var}" == "your_*_here" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing or invalid environment variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        log_info "Please update your .env file with valid API keys"
        exit 1
    fi
    
    log_success "Environment variables validated!"
}

test_docker_build() {
    log_info "Testing Docker build..."
    
    if docker build -t trading-bot-test . > /dev/null 2>&1; then
        log_success "Docker build successful!"
        docker rmi trading-bot-test > /dev/null 2>&1
    else
        log_error "Docker build failed!"
        log_info "Please fix any build errors before deploying"
        exit 1
    fi
}

deploy_railway() {
    log_info "Deploying to Railway..."
    
    # Check if Railway CLI is installed
    if ! command -v railway &> /dev/null; then
        log_info "Installing Railway CLI..."
        curl -fsSL https://railway.app/install.sh | sh
        export PATH="$HOME/.railway/bin:$PATH"
    fi
    
    # Login to Railway
    log_info "Please login to Railway when prompted..."
    railway login
    
    # Initialize project if needed
    if [[ ! -f "railway.toml" ]]; then
        log_info "Initializing Railway project..."
        railway link
    fi
    
    # Set environment variables
    log_info "Setting environment variables..."
    while IFS='=' read -r key value; do
        [[ $key =~ ^[[:space:]]*# ]] && continue
        [[ -z $key ]] && continue
        railway variables set "$key=$value"
    done < .env
    
    # Deploy
    log_info "Deploying to Railway..."
    railway up
    
    log_success "Deployment to Railway completed!"
    log_info "Check your Railway dashboard for the deployment URL"
}

deploy_render() {
    log_info "Deploying to Render..."
    
    # Create render.yaml if it doesn't exist
    if [[ ! -f "render.yaml" ]]; then
        log_info "Creating Render configuration..."
        cat > render.yaml << EOF
services:
  - type: web
    name: trading-bot
    env: docker
    dockerfilePath: ./Dockerfile
    region: oregon
    plan: starter
    envVars:
      - key: ENVIRONMENT
        value: ${ENVIRONMENT}
      - key: LOG_LEVEL
        value: INFO
EOF
    fi
    
    log_success "Render configuration created!"
    log_info "Please:"
    log_info "1. Push your code to GitHub"
    log_info "2. Go to render.com and create a new web service"
    log_info "3. Connect your GitHub repository"
    log_info "4. Render will automatically detect the render.yaml configuration"
    log_info "5. Add your environment variables in the Render dashboard"
}

deploy_digitalocean() {
    log_info "Deploying to DigitalOcean..."
    
    # Check if doctl is installed
    if ! command -v doctl &> /dev/null; then
        log_error "DigitalOcean CLI (doctl) not found!"
        log_info "Install it from: https://docs.digitalocean.com/reference/doctl/how-to/install/"
        exit 1
    fi
    
    # Check authentication
    if ! doctl account get > /dev/null 2>&1; then
        log_info "Please authenticate with DigitalOcean..."
        doctl auth init
    fi
    
    # Create App Platform spec
    cat > .do/app.yaml << EOF
name: trading-bot
services:
- name: web
  source_dir: /
  dockerfile_path: Dockerfile
  github:
    repo: $(git config --get remote.origin.url | sed 's/.*github\.com[:/]\([^/]*\/[^/]*\)\.git/\1/')
    branch: main
  instance_count: 1
  instance_size_slug: basic-xxs
  http_port: 5000
  health_check:
    http_path: /health
    port: 8081
  envs:
  - key: ENVIRONMENT
    value: ${ENVIRONMENT}
  - key: LOG_LEVEL
    value: INFO
EOF

    mkdir -p .do
    
    log_info "Creating DigitalOcean App..."
    doctl apps create --spec .do/app.yaml
    
    log_success "DigitalOcean App created!"
    log_info "Check your DigitalOcean dashboard for deployment status"
}

deploy_aws() {
    log_info "Deploying to AWS ECS..."
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found!"
        log_info "Install it from: https://aws.amazon.com/cli/"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        log_error "AWS credentials not configured!"
        log_info "Run: aws configure"
        exit 1
    fi
    
    AWS_REGION=${AWS_REGION:-us-east-1}
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REPOSITORY="trading-bot"
    
    log_info "Building and pushing Docker image to ECR..."
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION > /dev/null 2>&1 || \
        aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION
    
    # Get login token
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    
    # Build and push image
    docker build -t $ECR_REPOSITORY .
    docker tag $ECR_REPOSITORY:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest
    
    log_success "Docker image pushed to ECR!"
    log_info "Please create ECS service manually or use AWS Console/CloudFormation"
    log_info "Image URI: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest"
}

deploy_gcp() {
    log_info "Deploying to Google Cloud Run..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "Google Cloud CLI not found!"
        log_info "Install it from: https://cloud.google.com/sdk"
        exit 1
    fi
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
        log_info "Please authenticate with Google Cloud..."
        gcloud auth login
    fi
    
    # Set project if not set
    if [[ -z "${GOOGLE_CLOUD_PROJECT}" ]]; then
        log_info "Please set your Google Cloud project:"
        gcloud config set project $(gcloud projects list --format="value(projectId)" | head -n1)
    fi
    
    # Enable required APIs
    log_info "Enabling required APIs..."
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com
    
    # Deploy to Cloud Run
    log_info "Deploying to Cloud Run..."
    gcloud run deploy trading-bot \
        --source . \
        --platform managed \
        --region us-central1 \
        --allow-unauthenticated \
        --memory $MEMORY \
        --cpu $CPU \
        --max-instances 1 \
        --set-env-vars ENVIRONMENT=$ENVIRONMENT,LOG_LEVEL=INFO
    
    log_success "Deployment to Google Cloud Run completed!"
}

create_monitoring_dashboard() {
    log_info "Creating monitoring dashboard..."
    
    cat > monitoring/dashboard.json << EOF
{
  "dashboard": {
    "title": "Trading Bot Monitoring",
    "panels": [
      {
        "title": "System Health",
        "type": "stat",
        "targets": [
          {
            "expr": "trading_bot_cpu_usage_percent",
            "legendFormat": "CPU Usage %"
          },
          {
            "expr": "trading_bot_memory_usage_percent", 
            "legendFormat": "Memory Usage %"
          }
        ]
      },
      {
        "title": "Trading Metrics",
        "type": "graph",
        "targets": [
          {
            "expr": "trading_bot_portfolio_value",
            "legendFormat": "Portfolio Value"
          },
          {
            "expr": "trading_bot_active_positions",
            "legendFormat": "Active Positions"
          }
        ]
      }
    ]
  }
}
EOF

    mkdir -p monitoring
    log_success "Monitoring dashboard configuration created in monitoring/dashboard.json"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --cpu)
            CPU="$2"
            shift 2
            ;;
        --auto-scaling)
            AUTO_SCALING="true"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "🚀 Trading Bot Cloud Deployment Script"
    echo "======================================="
    
    if [[ -z "$PLATFORM" ]]; then
        log_error "Platform is required! Use --platform to specify one."
        show_help
        exit 1
    fi
    
    log_info "Deploying to: $PLATFORM"
    log_info "Environment: $ENVIRONMENT"
    log_info "Memory: $MEMORY"
    log_info "CPU: $CPU"
    log_info "Auto-scaling: $AUTO_SCALING"
    echo
    
    # Run pre-deployment checks
    check_prerequisites
    validate_env_variables
    test_docker_build
    
    # Deploy to specified platform
    case $PLATFORM in
        railway)
            deploy_railway
            ;;
        render)
            deploy_render
            ;;
        digitalocean)
            deploy_digitalocean
            ;;
        aws)
            deploy_aws
            ;;
        gcp)
            deploy_gcp
            ;;
        *)
            log_error "Unsupported platform: $PLATFORM"
            log_info "Supported platforms: railway, render, digitalocean, aws, gcp"
            exit 1
            ;;
    esac
    
    # Create monitoring setup
    create_monitoring_dashboard
    
    echo
    log_success "🎉 Deployment completed successfully!"
    log_info "Your trading bot should now be running in the cloud."
    log_info "Check the platform-specific dashboard for deployment status and logs."
    
    echo
    log_info "Next steps:"
    log_info "1. Verify the deployment is working: curl your-app-url/health"
    log_info "2. Monitor logs for any errors"
    log_info "3. Set up alerts and monitoring"
    log_info "4. Test with paper trading before going live"
    
    echo
    log_warning "⚠️  IMPORTANT: Always test thoroughly before using real money!"
}

# Run main function
main