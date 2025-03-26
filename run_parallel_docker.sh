#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Create final_exports directory if it doesn't exist
mkdir -p final_exports

# Set log file name
LOG_FILE="logs/docker_execution_log.txt"

# Start logging
echo "==== Docker Compose Execution Log - $(date) ====" > $LOG_FILE

# Check if the Python file exists in the current directory
if [ ! -f "Watermarked_Gen_Agent.py" ]; then
    echo "Error: Watermarked_Gen_Agent.py not found in the current directory." | tee -a $LOG_FILE
    exit 1
fi

# Check if source_pngs directory exists
if [ ! -d "source_pngs" ]; then
    echo "Error: source_pngs directory not found." | tee -a $LOG_FILE
    exit 1
fi

# Check if doc_content_gen directory exists
if [ ! -d "doc_content_gen" ]; then
    echo "Error: doc_content_gen directory not found in the current directory." | tee -a $LOG_FILE
    exit 1
fi

# Get number of CPU cores
CPU_CORES=$(nproc)
echo "Detected $CPU_CORES CPU cores." | tee -a $LOG_FILE

# Calculate number of containers (1 container per 4 cores)
NUM_CONTAINERS=$(($CPU_CORES / 4))
if [ $NUM_CONTAINERS -lt 1 ]; then
    NUM_CONTAINERS=1
    echo "Less than 4 cores detected, using 1 container." | tee -a $LOG_FILE
else
    echo "Creating $NUM_CONTAINERS containers (4 cores each)." | tee -a $LOG_FILE
fi

# Count total PNG files in source directory
TOTAL_FILES=$(find ./source_pngs -type f -name "*.png" | wc -l)
echo "Found $TOTAL_FILES PNG files to process." | tee -a $LOG_FILE

if [ $TOTAL_FILES -eq 0 ]; then
    echo "Error: No PNG files found in source_pngs directory." | tee -a $LOG_FILE
    exit 1
fi

# Create temporary directories for each container
echo "Creating temporary directories for file distribution..." | tee -a $LOG_FILE
for i in $(seq 1 $NUM_CONTAINERS); do
    mkdir -p "source_pngs_container_$i"
done

# Distribute files across containers
echo "Distributing files across containers..." | tee -a $LOG_FILE
COUNT=0
find ./source_pngs -type f -name "*.png" | while read file; do
    CONTAINER_INDEX=$((($COUNT % $NUM_CONTAINERS) + 1))
    cp "$file" "source_pngs_container_$CONTAINER_INDEX/"
    COUNT=$((COUNT + 1))
done


# Generate dynamic docker-compose.yml
echo "Generating docker-compose.yml for $NUM_CONTAINERS containers..." | tee -a $LOG_FILE
cat > docker-compose.yml << EOF
version: '3'

services:
EOF

for i in $(seq 1 $NUM_CONTAINERS); do
    cat >> docker-compose.yml << EOF
  jupyter-$i:
    image: jupyter-environment:latest
    container_name: jupyter-environment-instance-$i
    cpus: 4
    volumes:
      - ./Watermarked_Gen_Agent.py:/home/jupyter_user/documents/Watermarked_Gen_Agent.py
      - ./logs:/home/jupyter_user/logs
      - ./final_exports:/home/jupyter_user/documents/final_exports
      - ./source_pngs_container_$i:/home/jupyter_user/source_pngs
      - ./doc_content_gen:/home/jupyter_user/documents/doc_content_gen
    command: >
      bash -c "
        source /home/jupyter_user/conda/etc/profile.d/conda.sh &&
        conda activate jupyter_env &&
        mkdir -p /home/jupyter_user/documents &&
        cd /home/jupyter_user/documents &&
        rm -rf /home/jupyter_user/documents/doc_content_gen/mockups_agent_watermarked/bulk_pngLogos_dir/* &&
        cp -r /home/jupyter_user/source_pngs/* /home/jupyter_user/documents/doc_content_gen/mockups_agent_watermarked/bulk_pngLogos_dir/ &&
        python /home/jupyter_user/documents/Watermarked_Gen_Agent.py > /home/jupyter_user/logs/python_output_$i.log 2>&1
      "
    tty: true

EOF
done

# Run docker compose
echo "Starting Docker containers with docker compose..." | tee -a $LOG_FILE
docker compose up -d 2>&1 | tee -a $LOG_FILE

# Capture exit code
COMPOSE_EXIT_CODE=${PIPESTATUS[0]}

# Check if execution was successful
if [ $COMPOSE_EXIT_CODE -ne 0 ]; then
    echo "Docker Compose execution failed with exit code $COMPOSE_EXIT_CODE" | tee -a $LOG_FILE
    echo "See logs for details." | tee -a $LOG_FILE
    exit 1
fi

echo "Docker containers started successfully. Waiting for all containers to complete..." | tee -a $LOG_FILE

# Wait for all containers to finish
ALL_COMPLETED=0
while [ $ALL_COMPLETED -eq 0 ]; do
    ALL_COMPLETED=1
    for i in $(seq 1 $NUM_CONTAINERS); do
        CONTAINER_STATUS=$(docker inspect --format='{{.State.Status}}' jupyter-environment-instance-$i 2>/dev/null)
        if [ "$CONTAINER_STATUS" = "running" ]; then
            ALL_COMPLETED=0
            break
        fi
    done
    if [ $ALL_COMPLETED -eq 0 ]; then
        echo "Containers still running. Waiting 15 seconds..." | tee -a $LOG_FILE
        sleep 15
    fi
done

echo "All containers have completed execution." | tee -a $LOG_FILE

# Copy logs and results from all containers
echo "Copying results from all containers..." | tee -a $LOG_FILE
for i in $(seq 1 $NUM_CONTAINERS); do
    echo "Copying results from container $i..." | tee -a $LOG_FILE
    docker cp jupyter-environment-instance-$i:/home/jupyter_user/documents/final_exports/. ./final_exports/ | tee -a $LOG_FILE
    docker cp jupyter-environment-instance-$i:/home/jupyter_user/logs/python_output_$i.log ./logs/ | tee -a $LOG_FILE
done

# Clean up
echo "Cleaning up..." | tee -a $LOG_FILE
docker compose down >> $LOG_FILE 2>&1

# Remove temporary source directories
echo "Removing temporary source directories..." | tee -a $LOG_FILE
for i in $(seq 1 $NUM_CONTAINERS); do
    rm -rf "source_pngs_container_$i"
done

echo "All operations completed. Logs saved to $LOG_FILE and individual container logs in logs/ directory" | tee -a $LOG_FILE
