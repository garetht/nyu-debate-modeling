#!/bin/bash

# Lambda Labs CLI - A comprehensive tool for managing Lambda Labs instances
# Author: Generated CLI Tool
# Version: 1.0.0

set -euo pipefail

# Global variables
SCRIPT_NAME="$(basename "$0")"
VERSION="1.0.0"
REMOTE_HOME_DIR="/home/ubuntu/mars-arnesen-gh"
REMOTE_LOGS_DIR="$REMOTE_HOME_DIR/logs"

source "$(dirname "$0")/bash_scripts/colors.sh"

# Function to prompt for environment variable if not set
prompt_env_var() {
    local var_name="$1"
    local description="$2"
    local url="$3"

    if [[ -z "${!var_name:-}" ]]; then
        echo -e "${YELLOW}Environment variable ${var_name} is not set.${NC}"
        echo -e "${BLUE}${description}${NC}"
        echo -e "${BLUE}Create one here: ${url}${NC}"
        echo -n "Please enter the ${var_name}: "
        read -r value
        if [[ -z "$value" ]]; then
            log_error "${var_name} cannot be empty"
            exit 1
        fi
        export "${var_name}=$value"
    fi
}

# Function to validate required environment variables
validate_env_vars() {
    prompt_env_var "PEM_FILEPATH" "Path to the Lambda Labs pem file" "https://cloud.lambda.ai/ssh-keys"
    prompt_env_var "LAMBDA_LABS_API_KEY" "Lambda Labs API Key" "https://cloud.lambda.ai/api-keys/cloud-api"
    prompt_env_var "OPENAI_API_KEY" "OpenAI API Key" "https://platform.openai.com/api-keys"

    # Validate PEM file exists
    if [[ ! -f "$PEM_FILEPATH" ]]; then
        log_error "PEM file not found: $PEM_FILEPATH"
        exit 1
    fi

    # Check PEM file permissions (should be 600 or 400)
    local pem_perms
    pem_perms=$(stat -c %a "$PEM_FILEPATH" 2>/dev/null || stat -f %A "$PEM_FILEPATH" 2>/dev/null)
    if [[ "$pem_perms" != "600" && "$pem_perms" != "400" ]]; then
        log_error "PEM file has incorrect permissions: $pem_perms (should be 600 or 400)"
        log_error "Fix with: chmod 600 $PEM_FILEPATH"
        exit 1
    fi
}

# Function to get instance IP from Lambda Labs API
get_instance_ip() {
    local instance_id="$1"

    log_debug "Fetching instance IP for ID: $instance_id"

    local response
    response=$(curl -s -H "Authorization: Bearer $LAMBDA_LABS_API_KEY" \
        "https://cloud.lambda.ai/api/v1/instances/$instance_id")

    if [[ $? -ne 0 ]]; then
        log_error "Failed to fetch instance information from Lambda Labs API"
        exit 1
    fi

    # Extract IP address from JSON response
    local ip
    ip=$(echo "$response" | grep -o '"ip": "[^"]*' | cut -d'"' -f4)

    if [[ -z "$ip" ]]; then
        log_error "Could not extract IP address from API response"
        log_debug "API Response: $response"
        exit 1
    fi

    echo "$ip"
}

# Function to get first instance ID
get_first_instance_id() {
    log_debug "Fetching first instance ID"

    local response
    response=$(curl -s -H "Authorization: Bearer $LAMBDA_LABS_API_KEY" \
        "https://cloud.lambda.ai/api/v1/instances")

    if [[ $? -ne 0 ]]; then
        log_error "Failed to fetch instances from Lambda Labs API"
        exit 1
    fi

    # Extract first instance ID from JSON response
    local instance_id
    instance_id=$(echo "$response" | grep -o '"id": "[^"]*' | head -1 | cut -d'"' -f4)

    if [[ -z "$instance_id" ]]; then
        log_error "No instances found in your Lambda Labs account"
        exit 1
    fi

    echo "$instance_id"
}

# Function to get cached instance info or fetch fresh
get_cached_instance_info() {
    local cache_file="./tmp/lambda_instance_cache.json"
    local force_refresh="${1:-false}"

    # Create tmp directory if it doesn't exist
    mkdir -p ./tmp

    # Check if cache exists and is valid (not force refresh)
    if [[ "$force_refresh" == "false" && -f "$cache_file" ]]; then
        # Check if cache is less than 1 hour old
        local cache_age
        cache_age=$(date -r "$cache_file" +%s 2>/dev/null || stat -c %Y "$cache_file" 2>/dev/null)
        local current_time
        current_time=$(date +%s)
        local age_diff=$((current_time - cache_age))

        if [[ $age_diff -lt 3600 ]]; then  # 1 hour = 3600 seconds
            log_debug "Using cached instance info (age: ${age_diff}s)"
            cat "$cache_file"
            return 0
        else
            log_debug "Cache expired (age: ${age_diff}s), fetching fresh data"
        fi
    fi

    # Fetch fresh instance info
    log_debug "Fetching fresh instance info"
    local instance_id
    instance_id=$(get_first_instance_id)

    local ip
    ip=$(get_instance_ip "$instance_id")

    # Create cache entry
    local cache_data
    cache_data=$(cat << EOF
{
    "instance_id": "$instance_id",
    "ip": "$ip",
    "timestamp": $(date +%s)
}
EOF
)

    # Save to cache
    echo "$cache_data" > "$cache_file"
    log_debug "Cached instance info: $instance_id -> $ip"

    echo "$cache_data"
}

# Function to extract data from cache JSON
extract_from_cache() {
    local cache_data="$1"
    local field="$2"

    echo "$cache_data" | grep -o "\"$field\": \"[^\"]*" | cut -d'"' -f4
}

# Function to perform initial rsync to remote
perform_initial_rsync() {
    local local_path="."
    local remote_path="$REMOTE_HOME_DIR/"
    log_info "Performing initial rsync to remote..."

    # Call rsync-to-remote with default parameters
    cmd_rsync_to_remote_internal "$local_path" "$remote_path";
}

# Function to get the standard SSH environment setup command
get_ssh_env_setup() {
    echo "cd $1 && ./host_setup.sh && export OPENAI_API_KEY=$OPENAI_API_KEY && export INPUT_ROOT=$REMOTE_HOME_DIR/data/ && export SRC_ROOT=$REMOTE_HOME_DIR/"
}

# SSH command implementation
cmd_ssh() {
    local user="ubuntu"
    local port="22"
    local remote_path="$REMOTE_HOME_DIR"
    local retry_count=0
    local max_retries=1

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--remote-path)
                remote_path="$2"
                shift 2
                ;;
            -h|--help)
                cat << EOF
Usage: $SCRIPT_NAME ssh [OPTIONS]

Connect to the first Lambda Labs instance via SSH using ubuntu user on port 22.

OPTIONS:
    -r, --remote-path PATH  Remote directory to change to (default: $REMOTE_HOME_DIR)
    -h, --help              Show this help message

ENVIRONMENT VARIABLES:
    PEM_FILEPATH           Path to the Lambda Labs pem file
    LAMBDA_LABS_API_KEY    Lambda Labs API Key

EXAMPLES:
    $SCRIPT_NAME ssh
    $SCRIPT_NAME ssh -r /home/ubuntu/project

EOF
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                exit 1
                ;;
            *)
                log_error "Unknown argument: $1"
                exit 1
                ;;
        esac
    done

    # Validate environment variables
    validate_env_vars

    perform_initial_rsync

    while [[ $retry_count -le $max_retries ]]; do
        # Get instance info (cached or fresh)
        local force_refresh="false"
        if [[ $retry_count -gt 0 ]]; then
            force_refresh="true"
            log_warn "SSH connection failed, refreshing cache and retrying in 10 seconds..."
            sleep 10
        fi

        local cache_data
        cache_data=$(get_cached_instance_info "$force_refresh")

        local instance_id
        instance_id=$(extract_from_cache "$cache_data" "instance_id")

        local ip
        ip=$(extract_from_cache "$cache_data" "ip")

        log_info "Using instance: $instance_id at $ip"

        # Execute SSH command
        local ssh_exit_code
        local ssh_env_setup
        ssh_env_setup=$(get_ssh_env_setup "$remote_path")
        ssh -ti "$PEM_FILEPATH" -p "$port" "$user@$ip" "$ssh_env_setup && bash --login"
        ssh_exit_code=$?

        # Check exit code - 0 means success (including normal user exit)
        # 130 is SIGINT (Ctrl+C) which is also normal user exit
        if [[ $ssh_exit_code -eq 0 || $ssh_exit_code -eq 130 ]]; then
            exit 0
        else
            # Other exit codes indicate connection/authentication failures
            log_warn "SSH connection failed with exit code: $ssh_exit_code"
            ((retry_count++))
            if [[ $retry_count -gt $max_retries ]]; then
                log_error "Failed to connect after $max_retries retries"
                exit 1
            fi
        fi
    done
}

# Background task start command implementation
cmd_bg_task_start() {
    local user="ubuntu"
    local port="22"
    local remote_path="$REMOTE_HOME_DIR"
    local retry_count=0
    local max_retries=1
    local bg_command=""
    local task_name=""
    local parsing_command=false
    local timeout_ms=4000  # Default 4 seconds in milliseconds

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--remote-path)
                remote_path="$2"
                shift 2
                ;;
            -n|--name)
                task_name="$2"
                shift 2
                ;;
            -t|--timeout)
                timeout_ms="$2"
                # Validate timeout is a positive integer
                if ! [[ "$timeout_ms" =~ ^[0-9]+$ ]] || [[ "$timeout_ms" -lt 1 ]]; then
                    log_error "Timeout must be a positive integer in milliseconds"
                    exit 1
                fi
                shift 2
                ;;
            -h|--help)
                cat << EOF
Usage: $SCRIPT_NAME bg-task start [OPTIONS] -- COMMAND

Launch a background task on the first Lambda Labs instance with nohup.

OPTIONS:
    -r, --remote-path PATH  Remote directory to run command from (default: $REMOTE_HOME_DIR)
    -n, --name NAME         Task name for log file (default: auto-generated from timestamp)
    -t, --timeout MS        Timeout in milliseconds to wait for task startup (default: 5000)
    -h, --help              Show this help message

ARGUMENTS:
    --                      Separator before the command to run
    COMMAND                 Command to run in the background

ENVIRONMENT VARIABLES:
    PEM_FILEPATH           Path to the Lambda Labs pem file
    LAMBDA_LABS_API_KEY    Lambda Labs API Key
    OPENAI_API_KEY         OpenAI API Key

EXAMPLES:
    $SCRIPT_NAME bg-task start -- python ./run-debate --configuration="config.json"
    $SCRIPT_NAME bg-task start -n "training-job" -- python train.py --epochs 100
    $SCRIPT_NAME bg-task start -r /home/ubuntu/project -- ./run_experiment.sh
    $SCRIPT_NAME bg-task start -t 10000 -- python slow_startup.py

The task will be run with nohup and output will be logged to:
    $REMOTE_LOGS_DIR/\$TASK_NAME.log

The command will wait up to the specified timeout to detect immediate startup failures.

EOF
                exit 0
                ;;
            --)
                parsing_command=true
                shift
                break
                ;;
            -*)
                log_error "Unknown option: $1"
                exit 1
                ;;
            *)
                if [[ "$parsing_command" == true ]]; then
                    break
                else
                    log_error "Unknown argument: $1. Use -- to separate command."
                    exit 1
                fi
                ;;
        esac
    done

    # Check if we have a command to run
    if [[ $# -eq 0 ]]; then
        log_error "No command provided. Use -- to separate the command to run."
        echo "Example: $SCRIPT_NAME bg-task start -- python script.py"
        exit 1
    fi

    # Build the command string
    bg_command="$*"

    # Generate task name if not provided
    if [[ -z "$task_name" ]]; then
        task_name="task-$(date +%Y%m%d-%H%M%S)"
    fi

    # Validate environment variables
    validate_env_vars

    perform_initial_rsync

    while [[ $retry_count -le $max_retries ]]; do
        # Get instance info (cached or fresh)
        local force_refresh="false"
        if [[ $retry_count -gt 0 ]]; then
            force_refresh="true"
            log_warn "SSH connection failed, refreshing cache and retrying in 10 seconds..."
            sleep 10
        fi

        local cache_data
        cache_data=$(get_cached_instance_info "$force_refresh")

        local instance_id
        instance_id=$(extract_from_cache "$cache_data" "instance_id")

        local ip
        ip=$(extract_from_cache "$cache_data" "ip")

        log_info "Using instance: $instance_id at $ip"
        log_info "Starting background task: $bg_command"
        log_info "Task name: $task_name"
        log_info "Startup failure timeout: ${timeout_ms}ms"
        log_info "Logs will be written to: $REMOTE_LOGS_DIR/$task_name.log"

        # Convert timeout from milliseconds to seconds for sleep
        local timeout_seconds
        timeout_seconds=$(echo "scale=3; $timeout_ms / 1000" | bc -l 2>/dev/null || echo "$(($timeout_ms / 1000))")

        # Build the remote command with process monitoring
        local ssh_env_setup
        ssh_env_setup=$(get_ssh_env_setup "$remote_path")

        # Create a comprehensive remote command that:
        # 1. Sets up environment and creates log directory
        # 2. Starts the command in background with nohup
        # 3. Captures the PID
        # 4. Waits for the specified timeout
        # 5. Checks if process is still running
        # 6. Returns appropriate exit code
        local remote_command="$ssh_env_setup && mkdir -p $REMOTE_LOGS_DIR && {
            source .venv/bin/activate
            nohup $bg_command > $REMOTE_LOGS_DIR/$task_name.log 2>&1 &
            bg_pid=\$!
            sleep $timeout_seconds
            if kill -0 \$bg_pid 2>/dev/null; then
                echo \"Task started successfully with PID \$bg_pid\"
                exit 0
            else
                echo \"Task failed during startup. Check logs at $REMOTE_LOGS_DIR/$task_name.log\"
                echo \"Error output:\"
                if [ -f \"$REMOTE_LOGS_DIR/$task_name.log\" ]; then
                    tail -20 \"$REMOTE_LOGS_DIR/$task_name.log\"
                else
                    echo \"No log file found - task may have failed before logging started\"
                fi
                wait \$bg_pid 2>/dev/null || exit 1
            fi
        }"

        # Execute SSH command
        local ssh_exit_code
        ssh -ti "$PEM_FILEPATH" -p "$port" "$user@$ip" "$remote_command"
        ssh_exit_code=$?

        # Check exit code - 0 means success
        if [[ $ssh_exit_code -eq 0 ]]; then
            log_info "Background task started successfully"
            log_info "To view logs, run: ./$SCRIPT_NAME bg-task logs -f $task_name"
            exit 0
        else
            # Task failed during startup - error details already shown by remote command
            log_error "Background task failed to start"
            ((retry_count++))
            if [[ $retry_count -gt $max_retries ]]; then
                log_error "Failed to start task after $max_retries retries"
                exit 1
            fi
        fi
    done
}

# Logs command implementation
cmd_bg_task_logs() {
    local user="ubuntu"
    local port="22"
    local retry_count=0
    local max_retries=1
    local log_file=""
    local lines=50
    local follow=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--lines)
                lines="$2"
                # Validate lines is a positive integer
                if ! [[ "$lines" =~ ^[0-9]+$ ]] || [[ "$lines" -lt 1 ]]; then
                    log_error "Lines must be a positive integer"
                    exit 1
                fi
                shift 2
                ;;
            -f|--follow)
                follow=true
                shift
                ;;
            -h|--help)
                cat << EOF
Usage: $SCRIPT_NAME bg-task logs [OPTIONS] LOG_FILE

View logs from background tasks on the first Lambda Labs instance.

OPTIONS:
    -n, --lines NUMBER      Number of lines to show (default: 50)
    -f, --follow            Follow log file (like tail -f)
    -h, --help              Show this help message

ARGUMENTS:
    LOG_FILE               Name of the log file (without .log extension)
                          Use the task name from bg-task command

ENVIRONMENT VARIABLES:
    PEM_FILEPATH           Path to the Lambda Labs pem file
    LAMBDA_LABS_API_KEY    Lambda Labs API Key

EXAMPLES:
    $SCRIPT_NAME bg-task logs task-20240101-123456
    $SCRIPT_NAME bg-task logs -n 100 training-job
    $SCRIPT_NAME bg-task logs -f task-20240101-123456
    $SCRIPT_NAME bg-task logs -f -n 20 training-job

The logs command will look for files at:
    $REMOTE_LOGS_DIR/\$LOG_FILE.log

EOF
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                exit 1
                ;;
            *)
                if [[ -z "$log_file" ]]; then
                    log_file="$1"
                else
                    log_error "Multiple log files specified. Only one is allowed."
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # Check if log file is provided
    if [[ -z "$log_file" ]]; then
        log_error "No log file specified"
        echo "Example: $SCRIPT_NAME bg-task logs task-20240101-123456"
        exit 1
    fi

    # Validate environment variables
    validate_env_vars

    while [[ $retry_count -le $max_retries ]]; do
        # Get instance info (cached or fresh)
        local force_refresh="false"
        if [[ $retry_count -gt 0 ]]; then
            force_refresh="true"
            log_warn "SSH connection failed, refreshing cache and retrying in 10 seconds..."
            sleep 10
        fi

        local cache_data
        cache_data=$(get_cached_instance_info "$force_refresh")

        local instance_id
        instance_id=$(extract_from_cache "$cache_data" "instance_id")

        local ip
        ip=$(extract_from_cache "$cache_data" "ip")

        log_info "Using instance: $instance_id at $ip"
        log_info "Viewing logs from: $REMOTE_LOGS_DIR/$log_file.log"

        # Build the remote command based on follow flag
        local remote_command
        if [[ "$follow" == true ]]; then
            # Show last N lines and then follow
            remote_command="if [ -f \"$REMOTE_LOGS_DIR/$log_file.log\" ]; then tail -n $lines -f \"$REMOTE_LOGS_DIR/$log_file.log\"; else echo \"Log file not found: $REMOTE_LOGS_DIR/$log_file.log\"; exit 1; fi"
        else
            # Just show last N lines
            remote_command="if [ -f \"$REMOTE_LOGS_DIR/$log_file.log\" ]; then tail -n $lines \"$REMOTE_LOGS_DIR/$log_file.log\"; else echo \"Log file not found: $REMOTE_LOGS_DIR/$log_file.log\"; exit 1; fi"
        fi

        # Execute SSH command
        local ssh_exit_code
        ssh -ti "$PEM_FILEPATH" -p "$port" "$user@$ip" "$remote_command"
        ssh_exit_code=$?

        # Check exit code - 0 means success
        # 130 is SIGINT (Ctrl+C) which is also normal for -f mode
        if [[ $ssh_exit_code -eq 0 || $ssh_exit_code -eq 130 ]]; then
            exit 0
        else
            # Other exit codes indicate connection/authentication failures or file not found
            log_warn "SSH connection failed with exit code: $ssh_exit_code"
            ((retry_count++))
            if [[ $retry_count -gt $max_retries ]]; then
                log_error "Failed to view logs after $max_retries retries"
                exit 1
            fi
        fi
    done
}

# Background task list command implementation
cmd_bg_task_list() {
    local user="ubuntu"
    local port="22"

    validate_env_vars

    local cache_data
    cache_data=$(get_cached_instance_info "false")
    local ip
    ip=$(extract_from_cache "$cache_data" "ip")

    log_info "Listing Python background tasks on instance $ip"

    local remote_command="ps aux | grep 'python' | grep -v grep"
    ssh -ti "$PEM_FILEPATH" -p "$port" "$user@$ip" "$remote_command"
}

# Background task stop command implementation
cmd_bg_task_stop() {
    local user="ubuntu"
    local port="22"
    local pid=""

    if [[ $# -eq 0 ]]; then
        log_error "No PID provided for stop command"
        echo "Usage: $SCRIPT_NAME bg-task stop <PID>"
        exit 1
    fi
    pid="$1"

    validate_env_vars

    local cache_data
    cache_data=$(get_cached_instance_info "false")
    local ip
    ip=$(extract_from_cache "$cache_data" "ip")

    log_info "Stopping process with PID $pid on instance $ip"

    local remote_command="kill $pid"
    ssh -ti "$PEM_FILEPATH" -p "$port" "$user@$ip" "$remote_command"
}

# Background task command implementation
cmd_bg_task() {
    if [[ $# -eq 0 ]]; then
        log_error "No subcommand provided for bg-task"
        echo "Usage: $SCRIPT_NAME bg-task [start|logs|list|stop] ..."
        exit 1
    fi

    local subcommand="$1"
    shift

    case "$subcommand" in
        start)
            cmd_bg_task_start "$@"
            ;;
        logs)
            cmd_bg_task_logs "$@"
            ;;
        list)
            cmd_bg_task_list "$@"
            ;;
        stop)
            cmd_bg_task_stop "$@"
            ;;
        *)
            log_error "Unknown subcommand for bg-task: $subcommand"
            echo "Usage: $SCRIPT_NAME bg-task [start|logs|list|stop] ..."
            exit 1
            ;;
    esac
}

# Rsync to remote command implementation
cmd_rsync_to_remote() {
    local local_path="."
    local remote_path="$REMOTE_HOME_DIR/"
    local user="ubuntu"
    local port="22"
    local retry_count=0
    local max_retries=1

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -l|--local-path)
                local_path="$2"
                shift 2
                ;;
            -r|--remote-path)
                remote_path="$2"
                shift 2
                ;;
            -h|--help)
                cat << EOF
Usage: $SCRIPT_NAME rsync-to-remote [OPTIONS]

Synchronize files from local machine to the first Lambda Labs instance.

OPTIONS:
    -l, --local-path PATH   Local path to sync from (default: .) 
    -r, --remote-path PATH  Remote path to sync to (default: $REMOTE_HOME_DIR/)
    -h, --help              Show this help message

ENVIRONMENT VARIABLES:
    PEM_FILEPATH           Path to the Lambda Labs pem file
    LAMBDA_LABS_API_KEY    Lambda Labs API Key

EXAMPLES:
    $SCRIPT_NAME rsync-to-remote
    $SCRIPT_NAME rsync-to-remote -l ./src -r /home/ubuntu/project

EOF
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                exit 1
                ;;
            *)
                log_error "Unknown argument: $1"
                exit 1
                ;;
        esac
    done

    # Validate environment variables
    validate_env_vars

    # Call internal function and set flag on success
    if cmd_rsync_to_remote_internal "$local_path" "$remote_path"; then
        set_initial_rsync_flag
        exit 0
    else
        exit 1
    fi
}

# Internal rsync to remote function (used by both cmd_rsync_to_remote and perform_initial_rsync)
cmd_rsync_to_remote_internal() {
    local local_path="$1"
    local remote_path="$2"
    local user="ubuntu"
    local port="22"
    local retry_count=0
    local max_retries=1

    while [[ $retry_count -le $max_retries ]]; do
        # Get instance info (cached or fresh)
        local force_refresh="false"
        if [[ $retry_count -gt 0 ]]; then
            force_refresh="true"
            log_warn "Rsync failed, refreshing cache and retrying in 10 seconds..."
            sleep 10
        fi

        local cache_data
        cache_data=$(get_cached_instance_info "$force_refresh")

        local instance_id
        instance_id=$(extract_from_cache "$cache_data" "instance_id")

        local ip
        ip=$(extract_from_cache "$cache_data" "ip")

        log_info "Using instance: $instance_id at $ip"
        log_info "Syncing from $local_path to $user@$ip:$remote_path"

        # Build rsync command with hardcoded excludes and filters
        local rsync_cmd=(
            "rsync"
            "-avz"
            "--progress"
            "--exclude=/.git"
            "--filter=dir-merge,- .gitignore"
            "-e" "ssh -i $PEM_FILEPATH -p $port -o StrictHostKeyChecking=no"
            "$local_path"
            "$user@$ip:$remote_path"
        )

        # Execute rsync command
        log_debug "Executing: ${rsync_cmd[*]}"
        local rsync_exit_code
        "${rsync_cmd[@]}"
        rsync_exit_code=$?

        # Check exit code - 0 means success
        if [[ $rsync_exit_code -eq 0 ]]; then
            return 0
        else
            log_warn "Rsync command failed with exit code: $rsync_exit_code"
            ((retry_count++))
            if [[ $retry_count -gt $max_retries ]]; then
                log_error "Failed to sync after $max_retries retries"
                return 1
            fi
        fi
    done
}

# Rsync to host command implementation
cmd_rsync_to_host() {
    local remote_path="$REMOTE_HOME_DIR/"
    local local_path="."
    local user="ubuntu"
    local port="22"
    local retry_count=0
    local max_retries=1

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--remote-path)
                remote_path="$2"
                shift 2
                ;;
            -l|--local-path)
                local_path="$2"
                shift 2
                ;;
            -h|--help)
                cat << EOF
Usage: $SCRIPT_NAME rsync-to-host [OPTIONS]

Synchronize files from the first Lambda Labs instance to local machine.

OPTIONS:
    -r, --remote-path PATH  Remote path to sync from (default: $REMOTE_HOME_DIR/)
    -l, --local-path PATH   Local path to sync to (default: .) 
    -h, --help              Show this help message

ENVIRONMENT VARIABLES:
    PEM_FILEPATH           Path to the Lambda Labs pem file
    LAMBDA_LABS_API_KEY    Lambda Labs API Key

EXAMPLES:
    $SCRIPT_NAME rsync-to-host
    $SCRIPT_NAME rsync-to-host -r /home/ubuntu/project -l ./backup

EOF
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                exit 1
                ;;
            *)
                log_error "Unknown argument: $1"
                exit 1
                ;;
        esac
    done

    # Validate environment variables
    validate_env_vars

    while [[ $retry_count -le $max_retries ]]; do
        # Get instance info (cached or fresh)
        local force_refresh="false"
        if [[ $retry_count -gt 0 ]]; then
            force_refresh="true"
            log_warn "Rsync failed, refreshing cache and retrying in 10 seconds..."
            sleep 10
        fi

        local cache_data
        cache_data=$(get_cached_instance_info "$force_refresh")

        local instance_id
        instance_id=$(extract_from_cache "$cache_data" "instance_id")

        local ip
        ip=$(extract_from_cache "$cache_data" "ip")

        log_info "Using instance: $instance_id at $ip"
        log_info "Syncing from $user@$ip:$remote_path to $local_path"

        # Build rsync command with hardcoded excludes and filters
        local rsync_cmd=(
            "rsync"
            "-avz"
            "--progress"
            "--exclude=/.git"
            "--filter=dir-merge,- .gitignore"
            "--ignore-existing"
            "-e" "ssh -i $PEM_FILEPATH -p $port -o StrictHostKeyChecking=no"
            "$user@$ip:$remote_path"
            "$local_path"
        )

        # Execute rsync command
        log_debug "Executing: ${rsync_cmd[*]}"
        local rsync_exit_code
        "${rsync_cmd[@]}"
        rsync_exit_code=$?

        # Check exit code - 0 means success
        if [[ $rsync_exit_code -eq 0 ]]; then
            exit 0
        else
            log_warn "Rsync command failed with exit code: $rsync_exit_code"
            ((retry_count++))
            if [[ $retry_count -gt $max_retries ]]; then
                log_error "Failed to sync after $max_retries retries"
                exit 1
            fi
        fi
    done
}

# Main help function
show_help() {
    cat << EOF
Lambda Labs CLI - A comprehensive tool for managing Lambda Labs instances

Usage: $SCRIPT_NAME [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]

GLOBAL OPTIONS:
    -h, --help              Show this help message
    -v, --version           Show version information
    --debug                 Enable debug output

COMMANDS:
    ssh                     Connect to the first Lambda Labs instance via SSH
    bg-task                 Launch and manage background tasks on the first Lambda Labs instance
    rsync-to-remote         Synchronize files from local to remote instance
    rsync-to-host           Synchronize files from remote instance to local

ENVIRONMENT VARIABLES:
    PEM_FILEPATH           Path to the Lambda Labs pem file
                          Create here: https://cloud.lambda.ai/ssh-keys

    LAMBDA_LABS_API_KEY    Lambda Labs API Key
                          Create here: https://cloud.lambda.ai/api-keys/cloud-api

    OPENAI_API_KEY         OpenAI API Key
                          Create here: https://platform.openai.com/api-keys

EXAMPLES:
    # SSH into the first instance
    $SCRIPT_NAME ssh

    # SSH into the first instance and change to a specific directory
    $SCRIPT_NAME ssh -r /home/ubuntu/project

    # Launch a background task
    $SCRIPT_NAME bg-task start -- python ./run-debate --configuration="config.json"

    # View logs of a background task
    $SCRIPT_NAME bg-task logs <task_name>

    # List running background tasks
    $SCRIPT_NAME bg-task list

    # Stop a background task
    $SCRIPT_NAME bg-task stop <PID>

    # Sync current directory to first instance
    $SCRIPT_NAME rsync-to-remote

    # Sync from first instance to current directory
    $SCRIPT_NAME rsync-to-host

    # Get help for a specific command
    $SCRIPT_NAME ssh --help
    $SCRIPT_NAME bg-task --help
    $SCRIPT_NAME rsync-to-remote --help

For more information about each command, use:
    $SCRIPT_NAME COMMAND --help

EOF
}

# Main function
main() {
    # Parse global options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                echo "$SCRIPT_NAME version $VERSION"
                exit 0
                ;;
            --debug)
                export DEBUG=1
                shift
                ;;
            ssh)
                shift
                cmd_ssh "$@"
                exit $?
                ;;
            bg-task)
                shift
                cmd_bg_task "$@"
                exit $?
                ;;
            rsync-to-remote)
                shift
                cmd_rsync_to_remote "$@"
                exit $?
                ;;
            rsync-to-host)
                shift
                cmd_rsync_to_host "$@"
                exit $?
                ;;
            -*)
                log_error "Unknown global option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
            *)
                log_error "Unknown command: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # No command provided
    log_error "No command provided"
    show_help
    exit 1
}

# Run main function with all arguments
main "$@"
