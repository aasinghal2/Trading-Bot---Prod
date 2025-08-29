#!/bin/bash
# Trading Bot Monitoring Script
# Monitors the health and performance of the deployed trading bot

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HEALTH_URL=""
METRICS_URL=""
CHECK_INTERVAL=60  # seconds
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_DISK=90
LOG_FILE="logs/monitor.log"

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"; }

show_help() {
    cat << EOF
🔍 Trading Bot Monitoring Script

Usage: $0 [OPTIONS]

Options:
  --url URL               Base URL of your deployed trading bot
  --interval SECONDS      Check interval in seconds [default: 60]
  --cpu-threshold PERCENT CPU alert threshold [default: 80]
  --memory-threshold PERCENT Memory alert threshold [default: 85]
  --disk-threshold PERCENT Disk alert threshold [default: 90]
  --continuous            Run continuous monitoring
  --check-once            Run a single check and exit
  --help                  Show this help message

Examples:
  $0 --url https://your-bot.railway.app --check-once
  $0 --url https://your-bot.render.com --continuous --interval 30
  $0 --url http://your-droplet-ip:5000 --continuous

EOF
}

check_prerequisites() {
    # Check if curl is installed
    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not installed!"
        exit 1
    fi
    
    # Check if jq is installed (for JSON parsing)
    if ! command -v jq &> /dev/null; then
        log_warning "jq not found - JSON parsing will be limited"
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
}

validate_url() {
    if [[ -z "$HEALTH_URL" ]]; then
        log_error "Health URL is required!"
        log_info "Use --url to specify your deployment URL"
        exit 1
    fi
    
    # Test if URL is reachable
    if ! curl -s --max-time 10 "$HEALTH_URL" > /dev/null; then
        log_error "Cannot reach $HEALTH_URL"
        log_info "Please check if your deployment is running and accessible"
        exit 1
    fi
}

check_basic_health() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    log_info "[$timestamp] Checking basic health..."
    
    local response=$(curl -s --max-time 30 "$HEALTH_URL" 2>/dev/null)
    local http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$HEALTH_URL" 2>/dev/null)
    
    if [[ "$http_code" == "200" ]]; then
        log_success "✅ Health check passed (HTTP $http_code)"
        return 0
    else
        log_error "❌ Health check failed (HTTP $http_code)"
        return 1
    fi
}

check_detailed_health() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    log_info "[$timestamp] Checking detailed health..."
    
    local detailed_url="${HEALTH_URL%/health}/health/detailed"
    local response=$(curl -s --max-time 30 "$detailed_url" 2>/dev/null)
    local http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$detailed_url" 2>/dev/null)
    
    if [[ "$http_code" != "200" ]]; then
        log_error "❌ Detailed health check failed (HTTP $http_code)"
        return 1
    fi
    
    # Parse JSON response if jq is available
    if command -v jq &> /dev/null && [[ -n "$response" ]]; then
        local status=$(echo "$response" | jq -r '.status // "UNKNOWN"')
        local uptime=$(echo "$response" | jq -r '.uptime_seconds // 0')
        local cpu_usage=$(echo "$response" | jq -r '.system.cpu_usage_percent // 0')
        local memory_usage=$(echo "$response" | jq -r '.system.memory_usage_percent // 0')
        local disk_usage=$(echo "$response" | jq -r '.system.disk_usage_percent // 0')
        local portfolio_value=$(echo "$response" | jq -r '.trading.portfolio_value // 0')
        local active_positions=$(echo "$response" | jq -r '.trading.active_positions // 0')
        
        # Display metrics
        echo "📊 System Metrics:"
        echo "   Status: $status"
        echo "   Uptime: $(printf "%.0f" "$uptime") seconds"
        echo "   CPU Usage: $(printf "%.1f" "$cpu_usage")%"
        echo "   Memory Usage: $(printf "%.1f" "$memory_usage")%"
        echo "   Disk Usage: $(printf "%.1f" "$disk_usage")%"
        echo "   Portfolio Value: \$$(printf "%.2f" "$portfolio_value")"
        echo "   Active Positions: $active_positions"
        
        # Check thresholds and alert
        check_thresholds "$cpu_usage" "$memory_usage" "$disk_usage"
        
        return 0
    else
        log_warning "Cannot parse detailed metrics (jq not available or invalid JSON)"
        return 1
    fi
}

check_thresholds() {
    local cpu_usage=$1
    local memory_usage=$2
    local disk_usage=$3
    
    local alerts=()
    
    # Check CPU threshold
    if (( $(echo "$cpu_usage > $ALERT_THRESHOLD_CPU" | bc -l 2>/dev/null || echo "0") )); then
        alerts+=("🔥 HIGH CPU: ${cpu_usage}% (threshold: ${ALERT_THRESHOLD_CPU}%)")
    fi
    
    # Check memory threshold
    if (( $(echo "$memory_usage > $ALERT_THRESHOLD_MEMORY" | bc -l 2>/dev/null || echo "0") )); then
        alerts+=("🧠 HIGH MEMORY: ${memory_usage}% (threshold: ${ALERT_THRESHOLD_MEMORY}%)")
    fi
    
    # Check disk threshold
    if (( $(echo "$disk_usage > $ALERT_THRESHOLD_DISK" | bc -l 2>/dev/null || echo "0") )); then
        alerts+=("💾 HIGH DISK: ${disk_usage}% (threshold: ${ALERT_THRESHOLD_DISK}%)")
    fi
    
    # Send alerts if any
    if [[ ${#alerts[@]} -gt 0 ]]; then
        for alert in "${alerts[@]}"; do
            log_warning "$alert"
        done
        send_alert "Trading Bot Alert" "$(IFS=$'\n'; echo "${alerts[*]}")"
    fi
}

check_metrics() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    log_info "[$timestamp] Checking Prometheus metrics..."
    
    if [[ -z "$METRICS_URL" ]]; then
        log_warning "Metrics URL not set, skipping metrics check"
        return 1
    fi
    
    local response=$(curl -s --max-time 30 "$METRICS_URL" 2>/dev/null)
    local http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$METRICS_URL" 2>/dev/null)
    
    if [[ "$http_code" == "200" ]]; then
        log_success "✅ Metrics endpoint accessible"
        
        # Parse key metrics
        local portfolio_value=$(echo "$response" | grep "trading_bot_portfolio_value" | awk '{print $2}' | tail -1)
        local active_positions=$(echo "$response" | grep "trading_bot_active_positions" | awk '{print $2}' | tail -1)
        local uptime=$(echo "$response" | grep "trading_bot_uptime_seconds" | awk '{print $2}' | tail -1)
        
        if [[ -n "$portfolio_value" ]]; then
            echo "📈 Portfolio Value: \$$(printf "%.2f" "$portfolio_value")"
        fi
        if [[ -n "$active_positions" ]]; then
            echo "📊 Active Positions: $active_positions"
        fi
        if [[ -n "$uptime" ]]; then
            echo "⏱️ Uptime: $(printf "%.0f" "$uptime") seconds"
        fi
        
        return 0
    else
        log_error "❌ Metrics check failed (HTTP $http_code)"
        return 1
    fi
}

check_trading_activity() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    log_info "[$timestamp] Checking trading activity..."
    
    local status_url="${HEALTH_URL%/health}/status"
    local response=$(curl -s --max-time 30 "$status_url" 2>/dev/null)
    local http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$status_url" 2>/dev/null)
    
    if [[ "$http_code" == "200" && -n "$response" ]]; then
        if command -v jq &> /dev/null; then
            local app_status=$(echo "$response" | jq -r '.status // "UNKNOWN"')
            local last_updated=$(echo "$response" | jq -r '.last_updated // "unknown"')
            
            echo "🤖 Application Status: $app_status"
            echo "🕐 Last Updated: $last_updated"
            
            # Check component health
            local components=$(echo "$response" | jq -r '.components // {}' | jq -r 'to_entries[] | "\(.key): \(.value)"')
            if [[ -n "$components" ]]; then
                echo "🔧 Component Health:"
                echo "$components" | while read -r line; do
                    echo "   $line"
                done
            fi
        fi
        return 0
    else
        log_warning "Trading activity check failed or not available"
        return 1
    fi
}

send_alert() {
    local subject="$1"
    local message="$2"
    
    # Try to send Slack notification if webhook is available
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local payload=$(cat <<EOF
{
    "text": "$subject",
    "attachments": [
        {
            "color": "warning",
            "fields": [
                {
                    "title": "Alert Details",
                    "value": "$message",
                    "short": false
                },
                {
                    "title": "Timestamp",
                    "value": "$(date)",
                    "short": true
                }
            ]
        }
    ]
}
EOF
)
        curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" > /dev/null 2>&1
        log_info "📱 Alert sent to Slack"
    fi
    
    # Log alert
    log_warning "🚨 ALERT: $subject - $message"
}

run_single_check() {
    log_info "🔍 Running single health check..."
    echo "=================================="
    
    local all_passed=true
    
    # Basic health check
    if ! check_basic_health; then
        all_passed=false
    fi
    
    echo ""
    
    # Detailed health check
    if ! check_detailed_health; then
        all_passed=false
    fi
    
    echo ""
    
    # Metrics check
    if ! check_metrics; then
        all_passed=false
    fi
    
    echo ""
    
    # Trading activity check
    if ! check_trading_activity; then
        all_passed=false
    fi
    
    echo "=================================="
    
    if $all_passed; then
        log_success "🎉 All checks passed!"
        exit 0
    else
        log_error "❌ Some checks failed!"
        exit 1
    fi
}

run_continuous_monitoring() {
    log_info "🔄 Starting continuous monitoring..."
    log_info "Check interval: ${CHECK_INTERVAL} seconds"
    log_info "Press Ctrl+C to stop"
    echo ""
    
    # Set up signal handler for graceful shutdown
    trap 'log_info "Monitoring stopped by user"; exit 0' INT TERM
    
    while true; do
        # Run checks
        check_basic_health
        check_detailed_health
        check_metrics
        check_trading_activity
        
        echo ""
        log_info "Next check in ${CHECK_INTERVAL} seconds..."
        sleep "$CHECK_INTERVAL"
    done
}

# Parse command line arguments
CONTINUOUS_MODE=false
CHECK_ONCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --url)
            HEALTH_URL="$2/health"
            METRICS_URL="$2/metrics"
            shift 2
            ;;
        --interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        --cpu-threshold)
            ALERT_THRESHOLD_CPU="$2"
            shift 2
            ;;
        --memory-threshold)
            ALERT_THRESHOLD_MEMORY="$2"
            shift 2
            ;;
        --disk-threshold)
            ALERT_THRESHOLD_DISK="$2"
            shift 2
            ;;
        --continuous)
            CONTINUOUS_MODE=true
            shift
            ;;
        --check-once)
            CHECK_ONCE=true
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
    echo "🔍 Trading Bot Monitoring"
    echo "========================"
    
    # Check prerequisites
    check_prerequisites
    
    # Validate URL
    validate_url
    
    # Run appropriate mode
    if $CHECK_ONCE; then
        run_single_check
    elif $CONTINUOUS_MODE; then
        run_continuous_monitoring
    else
        log_error "Please specify either --check-once or --continuous"
        show_help
        exit 1
    fi
}

# Run main function
main