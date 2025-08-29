# 🚀 Cloud Deployment Guide

This guide covers deploying your AI-powered trading bot to the cloud for 24/7 operation. We'll cover multiple cloud platforms and deployment strategies.

## 📋 Quick Deployment Summary

| Platform | Cost (est.) | Setup Time | Best For |
|----------|-------------|------------|----------|
| **Railway** | $5-20/month | 5 minutes | Beginners, quick setup |
| **Render** | $7-25/month | 10 minutes | Simplicity, auto-scaling |
| **DigitalOcean** | $12-40/month | 15 minutes | Control, performance |
| **AWS ECS** | $15-50/month | 30 minutes | Enterprise, scalability |
| **Google Cloud Run** | $10-30/month | 20 minutes | Pay-per-use |

## 🎯 Pre-Deployment Checklist

### 1. Environment Setup
```bash
# 1. Copy environment template
cp env_template.txt .env

# 2. Fill in your API keys in .env
nano .env
```

### 2. Required API Keys
- **Alpha Vantage**: Free market data (5 calls/min)
- **OpenAI**: For AI analysis ($5-20/month)
- **Reddit/Twitter**: For sentiment analysis (free)
- **News API**: For news sentiment (free tier available)

### 3. Test Locally First
```bash
# Build and test Docker container
docker-compose up --build

# Test the application
curl http://localhost:8081/health
```

---

## 🚄 Option 1: Railway (Recommended for Beginners)

**Pros**: Extremely simple, automatic deployments, built-in monitoring
**Cons**: Limited customization, can be expensive for high usage

### Step-by-Step Deployment

1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Deploy from GitHub**
   ```bash
   # Push your code to GitHub first
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/trading-bot.git
   git push -u origin main
   ```

3. **Create Railway Project**
   - Click "Deploy from GitHub repo"
   - Select your trading bot repository
   - Railway will auto-detect the Dockerfile

4. **Configure Environment Variables**
   ```bash
   # In Railway dashboard, go to Variables tab and add:
   ALPHA_VANTAGE_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   REDDIT_CLIENT_ID=your_id_here
   REDDIT_CLIENT_SECRET=your_secret_here
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   ```

5. **Configure Resources**
   - Memory: 2GB (for ML libraries)
   - CPU: 1 vCPU
   - Enable auto-deploy on git push

6. **Monitor Deployment**
   - Check logs in Railway dashboard
   - Your bot will be running 24/7!

### Railway Pricing
- **Starter**: $5/month (512MB RAM)
- **Pro**: $20/month (8GB RAM) - Recommended

---

## 🎨 Option 2: Render

**Pros**: Free tier available, great for prototyping
**Cons**: Free tier has limitations, can be slow

### Deployment Steps

1. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Connect your GitHub account

2. **Create Web Service**
   - New > Web Service
   - Connect your repository
   - Use these settings:
     ```
     Build Command: docker build -t trading-bot .
     Start Command: docker run -p 10000:5000 trading-bot
     ```

3. **Environment Variables**
   Add all your API keys in the Environment tab

4. **Configure Auto-Deploy**
   - Enable auto-deploy on git push
   - Set branch to `main`

### Render Pricing
- **Free**: Limited hours, sleeps after 15min inactivity
- **Starter**: $7/month - Always on, 512MB RAM
- **Standard**: $25/month - 2GB RAM (Recommended)

---

## 🌊 Option 3: DigitalOcean Droplet

**Pros**: Full control, predictable pricing, good performance
**Cons**: Requires more setup and maintenance

### Step-by-Step Setup

1. **Create Droplet**
   ```bash
   # Recommended specs:
   # - Ubuntu 22.04 LTS
   # - 2GB RAM / 1 vCPU ($12/month)
   # - Enable monitoring and backups
   ```

2. **Initial Server Setup**
   ```bash
   # SSH into your droplet
   ssh root@your-droplet-ip

   # Update system
   apt update && apt upgrade -y

   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh

   # Install Docker Compose
   apt install docker-compose-plugin -y

   # Create app user
   useradd -m -s /bin/bash trading
   usermod -aG docker trading
   ```

3. **Deploy Application**
   ```bash
   # Switch to trading user
   su - trading

   # Clone your repository
   git clone https://github.com/yourusername/trading-bot.git
   cd trading-bot

   # Create .env file
   cp env_template.txt .env
   nano .env  # Add your API keys

   # Start the application
   docker-compose up -d
   ```

4. **Set Up Reverse Proxy (Optional)**
   ```bash
   # Install Nginx
   sudo apt install nginx -y

   # Configure Nginx
   sudo nano /etc/nginx/sites-available/trading-bot
   ```

   Add this configuration:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }

       location /health {
           proxy_pass http://localhost:8081;
       }
   }
   ```

5. **Enable Auto-Start**
   ```bash
   # Create systemd service
   sudo nano /etc/systemd/system/trading-bot.service
   ```

   ```ini
   [Unit]
   Description=Trading Bot
   Requires=docker.service
   After=docker.service

   [Service]
   Type=oneshot
   RemainAfterExit=yes
   WorkingDirectory=/home/trading/trading-bot
   ExecStart=/usr/bin/docker-compose up -d
   ExecStop=/usr/bin/docker-compose down
   User=trading

   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   # Enable the service
   sudo systemctl enable trading-bot
   sudo systemctl start trading-bot
   ```

### DigitalOcean Pricing
- **Basic**: $12/month (2GB RAM, 1 vCPU)
- **General Purpose**: $18/month (2GB RAM, 1 vCPU, dedicated CPU)

---

## ☁️ Option 4: AWS ECS Fargate

**Pros**: Highly scalable, enterprise-grade, AWS ecosystem
**Cons**: Complex setup, can be expensive

### Prerequisites
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
```

### Deployment Steps

1. **Create ECR Repository**
   ```bash
   # Create repository
   aws ecr create-repository --repository-name trading-bot

   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
   ```

2. **Build and Push Image**
   ```bash
   # Build image
   docker build -t trading-bot .

   # Tag for ECR
   docker tag trading-bot:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/trading-bot:latest

   # Push to ECR
   docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/trading-bot:latest
   ```

3. **Create ECS Resources**
   ```bash
   # Create cluster
   aws ecs create-cluster --cluster-name trading-bot-cluster

   # Create task definition (see aws-task-definition.json below)
   aws ecs register-task-definition --cli-input-json file://aws-task-definition.json

   # Create service
   aws ecs create-service \
     --cluster trading-bot-cluster \
     --service-name trading-bot-service \
     --task-definition trading-bot:1 \
     --desired-count 1 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
   ```

4. **Task Definition** (`aws-task-definition.json`)
   ```json
   {
     "family": "trading-bot",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "executionRoleArn": "arn:aws:iam::123456789:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "trading-bot",
         "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/trading-bot:latest",
         "portMappings": [
           {
             "containerPort": 5000,
             "protocol": "tcp"
           },
           {
             "containerPort": 8081,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {"name": "ENVIRONMENT", "value": "production"},
           {"name": "LOG_LEVEL", "value": "INFO"}
         ],
         "secrets": [
           {"name": "OPENAI_API_KEY", "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:trading-bot-secrets:OPENAI_API_KEY"},
           {"name": "ALPHA_VANTAGE_API_KEY", "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:trading-bot-secrets:ALPHA_VANTAGE_API_KEY"}
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/trading-bot",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

### AWS Pricing (Estimated)
- **Fargate**: $15-50/month depending on usage
- **ECR**: $1/month for image storage
- **CloudWatch**: $5/month for logs
- **Secrets Manager**: $0.40/secret/month

---

## 🔧 Option 5: Google Cloud Run

**Pros**: Pay-per-use, automatic scaling, generous free tier
**Cons**: Cold starts, request-based pricing model

### Deployment Steps

1. **Setup Google Cloud**
   ```bash
   # Install gcloud CLI
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```

2. **Build and Deploy**
   ```bash
   # Set project
   gcloud config set project your-project-id

   # Enable required APIs
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com

   # Build and deploy
   gcloud run deploy trading-bot \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 1 \
     --max-instances 1 \
     --set-env-vars ENVIRONMENT=production
   ```

3. **Configure Secrets**
   ```bash
   # Create secrets
   echo -n "your-openai-key" | gcloud secrets create openai-api-key --data-file=-
   echo -n "your-alpha-vantage-key" | gcloud secrets create alpha-vantage-key --data-file=-

   # Update service with secrets
   gcloud run services update trading-bot \
     --update-secrets OPENAI_API_KEY=openai-api-key:latest \
     --update-secrets ALPHA_VANTAGE_API_KEY=alpha-vantage-key:latest
   ```

### Google Cloud Pricing
- **Free Tier**: 2 million requests/month
- **Paid**: $0.0000024 per 100ms of CPU time
- **Estimated**: $10-30/month for continuous operation

---

## 📊 Monitoring and Alerts

### Health Check Endpoint
Your bot includes a health check endpoint at `/health` (port 8081):

```bash
# Test health check
curl http://your-deployment-url:8081/health
```

### Log Monitoring
```bash
# View logs (Docker Compose)
docker-compose logs -f trading-bot

# View logs (systemd)
sudo journalctl -u trading-bot -f
```

### Set Up Alerts

1. **Slack Notifications**
   ```bash
   # Add to your .env file
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
   ```

2. **Email Alerts**
   ```bash
   # Configure email in .env
   EMAIL_NOTIFICATIONS=true
   EMAIL_SMTP_SERVER=smtp.gmail.com
   EMAIL_USERNAME=your-email@gmail.com
   EMAIL_PASSWORD=your-app-password
   ```

3. **Uptime Monitoring**
   Use services like:
   - **UptimeRobot** (free)
   - **Pingdom** (paid)
   - **DataDog** (enterprise)

---

## 💰 Cost Optimization Tips

### 1. Resource Right-Sizing
```yaml
# Optimize Docker resource limits
deploy:
  resources:
    limits:
      memory: 1.5G  # Reduce if possible
      cpus: '0.8'   # Monitor CPU usage
```

### 2. Trading Schedule
```python
# Only trade during market hours
TRADING_HOURS = {
    'start': '09:30',  # Market open
    'end': '16:00',    # Market close
    'timezone': 'US/Eastern'
}
```

### 3. API Usage Optimization
- Use free tiers: Alpha Vantage (5 calls/min)
- Cache data when possible
- Batch API requests
- Use webhooks instead of polling

### 4. Storage Optimization
```bash
# Clean up old logs
find logs/ -name "*.log" -mtime +7 -delete

# Compress old data
gzip data/old_*.json
```

---

## 🚨 Security Best Practices

### 1. Environment Variables
```bash
# Never commit .env files
echo ".env" >> .gitignore

# Use secrets management in production
# AWS: Secrets Manager
# GCP: Secret Manager
# Azure: Key Vault
```

### 2. Network Security
```bash
# Firewall rules (DigitalOcean example)
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
```

### 3. Container Security
```dockerfile
# Run as non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser
```

### 4. API Key Rotation
- Rotate API keys monthly
- Use separate keys for different environments
- Monitor API usage for anomalies

---

## 🔄 Continuous Deployment

### GitHub Actions (Automatic Deployment)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Trading Bot

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Railway
      uses: railway/deploy@v1
      with:
        token: ${{ secrets.RAILWAY_TOKEN }}
        service: trading-bot
```

### Auto-Restart on Failure
```bash
# Docker Compose with restart policy
restart: unless-stopped

# Systemd service with restart
Restart=always
RestartSec=10
```

---

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Monitor memory usage
   docker stats trading-bot
   
   # Increase memory limits
   memory: 3G  # In docker-compose.yml
   ```

2. **API Rate Limits**
   ```bash
   # Check logs for rate limit errors
   grep "rate limit" logs/trading_system.log
   
   # Increase intervals in config
   real_time_interval: 5  # Seconds between API calls
   ```

3. **Market Hours**
   ```bash
   # Ensure your bot respects market hours
   # Check timezone settings in config.yaml
   ```

### Debug Mode
```bash
# Run with debug logging
docker-compose up -d
docker-compose logs -f trading-bot

# Or run locally
python main.py --mode auto --verbose
```

---

## 📞 Support

If you need help with deployment:

1. **Check logs first**: Most issues are visible in application logs
2. **Verify API keys**: Ensure all required keys are set
3. **Test locally**: Always test with Docker Compose first
4. **Monitor resources**: Check CPU/memory usage
5. **Join communities**: Reddit r/algotrading, Discord servers

---

## 🎉 Deployment Checklist

- [ ] API keys configured and working
- [ ] Docker container builds successfully
- [ ] Health check endpoint responds
- [ ] Trading configuration is correct
- [ ] Monitoring and alerts set up
- [ ] Backup strategy implemented
- [ ] Security measures in place
- [ ] Cost optimization applied
- [ ] Documentation updated

**Congratulations! Your trading bot is now running 24/7 in the cloud! 🚀**

---

*Remember: Always start with paper trading to test your strategies before using real money. Trading involves risk, and past performance doesn't guarantee future results.*