import torch
from inspect_ai.model import get_model, GenerateConfig

OUTPUT_CONFIG = dict(
    display="plain",
    log_dir="outputs/inspect_logs",
    log_format="json",
)

OLLAMA_QWEN_MODEL = get_model(
    model="ollama/qwen2.5:1.5b"
)


def create_local_llama_model(max_connections: int):
    return get_model(
        "hf/local",
        model_path="/home/ubuntu/mars-arnesen-gh/garethtan/downloaded-models/gradientai/Llama-3-8B-Instruct-262k",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=GenerateConfig(
            max_connections=max_connections
        )
    )


GPT_O4_MINI_MODEL = get_model(
    model="openai/o4-mini-2025-04-16",
    config=GenerateConfig(
        max_connections=8
    )
)

DEBATER_LOCAL_EVAL_CONFIG = dict(
    model=[
        OLLAMA_QWEN_MODEL,
    ],
    max_connections=1,
)

# DEBATER_LOJBAN_EVAL_CONFIG = dict(
#     model=[
#         create_local_llama_model(max_connections=2),
#         GPT_O4_MINI_MODEL,
#     ],
# )
#
# DEBATER_QUALITY_EVAL_CONFIG = dict(
#     model=[
#         create_local_llama_model(max_connections=4),
#         GPT_O4_MINI_MODEL,
#     ],
# )


GPT_4_1_MODEL_CONFIG = get_model(
    model="openai/gpt-4.1-2025-04-14",
    config=GenerateConfig(
        max_connections=8
    )
)

GPT_4_1_NANO_MODEL_CONFIG = get_model(
    model="openai/gpt-4.1-nano-2025-04-14",
    config=GenerateConfig(
        max_connections=10,
        temperature=0.5
    )
)


GPT_4_1_NANO_FINETUNED_MODEL_CONFIG = get_model(
    model="openai/ft:gpt-4.1-nano-2025-04-14:modulo-research-ltd:michael-and-khan-data-judge-nano-16-09:CGRIpeD6",
    config=GenerateConfig(
        max_connections=10,
        temperature=0.5
    )
)

JUDGE_EVAL_CONFIG = dict(
    model=[
        # GPT_4_1_MODEL_CONFIG,
        # GPT_4_1_NANO_MODEL_CONFIG
        # GPT_4_1_NANO_FINETUNED_MODEL_CONFIG
        create_local_llama_model(max_connections=4)
    ]
)

