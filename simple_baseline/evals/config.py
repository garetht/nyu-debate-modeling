CONFIG = dict(
    model="hf/local",
    max_connections=1,
    model_args={
        "model_path": "/home/ubuntu/mars-arnesen-gh/garethtan/downloaded-models/gradientai/Llama-3-8B-Instruct-262k",
    },
    display="plain",
    log_dir="outputs/inspect_logs",
    log_format="json",
)

# CONFIG = dict(
#     model="ollama/qwen2.5:1.5b"
# )
