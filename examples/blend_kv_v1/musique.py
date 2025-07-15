# Standard
from dataclasses import asdict
import argparse
import contextlib
import os
import time
import json
import numpy as np

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder

# Local imports
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1


def setup_environment_variables(
    use_disk: bool = False, blend_special_str: str = " # # "
):
    """Setup LMCache environment variables for blending."""
    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"

    if use_disk:
        # Disable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "False"

        # Set the maximum size of the local CPU buffer size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

        # Enable local disk backend in LMCache
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"

        # Set the maximum size of the local disk size to 10GB
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        # Enable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "True"

        # Set the maximum size of the local CPU size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

    os.environ["ENABLE_METRICS"] = "True"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    """Build LLM with LMCache integration."""
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm = LLM(
        model=model,
        kv_transfer_config=ktc, # CacheBlend
        max_model_len=8000,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        enforce_eager=True # disable torch compile
        # tensor_parallel_size=4 # use more than one gpu for big models
    )

    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)

def create_blended_prompt(tokenizer, contexts, question, blend_special_str):
    # System prompt (includes BOS token)
    sys_prompt = tokenizer.encode(
        "You will be asked a question after reading several passages. "
        "Please directly answer the question based on the given passages. "
        "Do NOT repeat the question. The answer should be within 5 words."
    )

    # Encode the blend separator (remove BOS token)
    blend_separator = tokenizer.encode(blend_special_str, add_special_tokens=False)
    
    # Encode each context as a separate chunk
    context_chunks = []
    for ctx in contexts:
        ctx_text = ctx['text']
        # Encode and remove BOS token
        ctx_tokens = tokenizer.encode(ctx_text, add_special_tokens=False)
        context_chunks.append(ctx_tokens)
        
        print(f"Context chunk length: {len(ctx_tokens)}")
    
    # Build the complete prompt
    prompt_tokens = sys_prompt
    
    # Add each context chunk with separators
    for ctx_chunk in context_chunks:
        prompt_tokens = prompt_tokens + blend_separator + ctx_chunk
    
    # Add the question with separator
    question_text = f"\n\nAnswer the question directly based on the given passages. " \
                   f"Do NOT repeat the question. The answer should be within 5 words.\n" \
                   f"Question: {normalize_question(question)}\nAnswer:"
    question_tokens = tokenizer.encode(question_text,add_special_tokens=False)
    print("final question")
    print(question_text)
    print(len(question_tokens))
    print(len(blend_separator))
    prompt_tokens = prompt_tokens + blend_separator + question_tokens
    
    print(f"Total prompt length: {len(prompt_tokens)}")
    
    return (prompt_tokens)


def run_generation_with_timing(llm, prompt_tokens, sampling_params, description):
    """Run generation and measure timing."""
    print(f"\n--- Running {description} ---")
    print(f"Prompt length: {len(prompt_tokens)} tokens")
    
    try:
        start_time = time.time()
        outputs = llm.generate(prompt_token_ids=prompt_tokens, sampling_params=sampling_params)
        end_time = time.time()
        
        generation_time = end_time - start_time
        generated_text = outputs[0].outputs[0].text
        
        print(f"Generated answer: {generated_text!r}")
        print(f"Generation time: {generation_time:.3f} seconds")
        
        return generated_text, generation_time
        
    except Exception as e:
        print(f"Error during generation: {e}")
        print(f"Prompt token count: {len(prompt_tokens)}")
        raise


def create_reordered_contexts(contexts):
    """Create two different orderings of the same contexts to demonstrate cache blending."""
    import random
    
    # Create a copy and shuffle it to get a different order
    reordered_contexts = contexts.copy()
    random.seed(42)
    random.shuffle(reordered_contexts)
    
    return reordered_contexts


def parse_args():
    parser = argparse.ArgumentParser(description="Demonstrate cache blending dataset")
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Specify whether to use disk as backend (default: False)",
    )
    parser.add_argument(
        "-b",
        "--blend-special-str",
        default="# #",
        help="Specify the special separators to separate chunks (default: '# #')",
    )
    parser.add_argument(
        "--dataset-path",
        default="musique_s.json",
        help="Path to dataset (default: musique_s.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configuration
    lmcache_connector = "LMCacheConnectorV1"
    # model = "mistralai/Mistral-7B-Instruct-v0.2"
    # model = "mistralai/Mixtral-8x7B-Instruct-v0.1" # doesn't work
    model = "meta-llama/Llama-3.1-8B-Instruct" # nonsense tokenizer
    # model = "meta-llama/Meta-Llama-3-8B-Instruct" # better but still kind of nonsense
    # model = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # model too big
    
    # Setup environment
    setup_environment_variables(args.use_disk, args.blend_special_str)
    
    # Load dataset and tokenizer
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Select a single example to demonstrate cache blending
    example = dataset[0]
    print(f"Selected example for cache blending demonstration")
    print(f"Question: {example['question']}")
    print(f"Expected answers: {example['answers']}")
    print(f"Number of contexts: {len(example['ctxs'])}")
    
    # Create two different orderings of the same contexts
    original_contexts = example["ctxs"]
    reordered_contexts = create_reordered_contexts(original_contexts)
    # reordered_contexts = original_contexts
    # original_contexts.pop(5) # remove middle context and see effect on kvcache hit
    
    # Track results
    first_time = 0.0
    second_time = 0.0
    f1_scores = []
    first_answer = ""
    second_answer = ""
    
    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=32)
        
        print(f"\n{'='*60}")
        print("FIRST REQUEST - Original context order (establishes cache)")
        print(f"{'='*60}")
        
        # Create blended prompt with original context order
        first_prompt_tokens = create_blended_prompt(
            tokenizer, 
            original_contexts, 
            example["question"][:-2], 
            args.blend_special_str
        )
        
        # Run first generation (establishes cache)
        first_answer, first_time = run_generation_with_timing(
            llm, first_prompt_tokens, sampling_params, "First request (original order)"
        )
        
        # Calculate F1 score for first answer
        f1_first = max([compute_f1(first_answer, answer, tokenizer) 
                        for answer in example["answers"]])
        f1_scores.append(f1_first)
        print(f"F1 Score: {f1_first:.3f}")
        
        # Add a small delay between requests
        time.sleep(1)
        
        print(f"\n{'='*60}")
        print("SECOND REQUEST - Reordered contexts (benefits from cache blending)")
        print(f"{'='*60}")
        
        # Create blended prompt with reordered contexts
        second_prompt_tokens = create_blended_prompt(
            tokenizer, 
            reordered_contexts, 
            # original_contexts,
            example["question"], 
            args.blend_special_str
        )
        
        # Run second generation (should benefit from cache blending)
        second_answer, second_time = run_generation_with_timing(
            llm, second_prompt_tokens, sampling_params, "Second request (reordered contexts)"
        )
        
        # Calculate F1 score for second answer
        f1_second = max([compute_f1(second_answer, answer, tokenizer) 
                        for answer in example["answers"]])
        f1_scores.append(f1_second)
        print(f"F1 Score: {f1_second:.3f}")
    
    # Print summary results
    print(f"\n{'='*60}")
    print("CACHE BLENDING DEMONSTRATION RESULTS")
    print(f"{'='*60}")
    print(f"First request time (original order): {first_time:.3f} seconds")
    print(f"Second request time (reordered contexts): {second_time:.3f} seconds")
    
    if second_time < first_time:
        speedup = (first_time - second_time) / first_time * 100
        print(f"\n🚀 CACHE BLENDING SPEEDUP: {speedup:.1f}% faster!")
        print(f"Time saved: {first_time - second_time:.3f} seconds")
    else:
        print(f"\nNo significant speedup observed (may need more contexts or larger chunks)")
    
    print(f"\nF1 Scores:")
    print(f"  First request: {f1_scores[0]:.3f}")
    print(f"  Second request: {f1_scores[1]:.3f}")
    
    print(f"\nBlend separator used: '{args.blend_special_str}'")
    print(f"Chunk size: {os.environ.get('LMCACHE_CHUNK_SIZE', 'default')} tokens")

if __name__ == "__main__":
    main()