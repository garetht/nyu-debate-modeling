#!/bin/bash

# Check if API key is set
if [ -z "$LAMBDA_LABS_API_KEY" ]; then
    echo "Error: LAMBDA_LABS_API_KEY environment variable is not set"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq first."
    exit 1
fi

# Function to make the API call and check for GPU availability
make_api_call() {
    echo "$(date): Checking GPU availability..."

    response=$(curl -s -X GET "https://cloud.lambda.ai/api/v1/instance-types" \
                    -H "accept: application/json" \
                    -H "Authorization: Bearer $LAMBDA_LABS_API_KEY")

    # Check if the response is valid JSON and contains the expected structure
    if ! echo "$response" | jq empty 2>/dev/null; then
        echo "Error: Invalid JSON response"
        echo "Response: $response"
        echo "---"
        return
    fi

    # Check if gpu_1x_gh200 exists in the response
    if ! echo "$response" | jq -e '.data.gpu_1x_gh200' >/dev/null 2>&1; then
        echo "Error: gpu_1x_gh200 not found in response"
        echo "Available instance types:"
        echo "$response" | jq -r '.data | keys[]' 2>/dev/null || echo "Could not parse instance types"
        echo "---"
        return
    fi

    # Check if gpu_1x_gh200 has available regions
    available_regions=$(echo "$response" | jq -r '.data.gpu_1x_gh200.regions_with_capacity_available | length' 2>/dev/null)

    if [ "$available_regions" -gt 0 ] 2>/dev/null; then
        regions=$(echo "$response" | jq -r '.data.gpu_1x_gh200.regions_with_capacity_available | if type == "array" then join(", ") else . end' 2>/dev/null)
        echo "🎉 GPU AVAILABLE! gpu_1x_gh200 is available in regions: $regions"

        # Show the full response for this instance type
        echo "Instance details:"
        echo "$response" | jq '.data.gpu_1x_gh200' 2>/dev/null || echo "Could not parse instance details"

        echo "🔗 Attempting to launch an instance (https://cloud.lambda.ai/instances)"
        curl -X POST "https://cloud.lambda.ai/api/v1/instance-operations/launch" \
         -H 'accept: application/json' \
         -H 'content-type: application/json' \
         -u "$LAMBDA_LABS_API_KEY:" \
         -d '{"region_name":"us-east-3","instance_type_name":"gpu_1x_gh200","ssh_key_names":["lambda-labs"],"file_system_names":["mars-arnesen-gh"],"file_system_mounts":[{"mount_point":"/lambda/nfs/mars-arnesen-gh","file_system_id":"3df25572f4c04cefb9928c6086c25794"}]}'

        exit 0
    else
        echo "No gpu_1x_gh200 capacity available"
    fi

    echo "---"
}

# Main loop
echo "Starting Lambda Labs GPU availability monitor (every 4 seconds)"
echo "Press Ctrl+C to stop"
echo "==="

while true; do
    make_api_call
    sleep 4
done
