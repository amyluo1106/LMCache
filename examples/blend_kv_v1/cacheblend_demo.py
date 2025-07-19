# Standard
from dataclasses import asdict
import argparse
import contextlib
import os
import time
import json
import numpy as np
import logging

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.utils import CacheEngineKey

# Local imports
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cacheblend_debug")

def setup_environment_variables(
    use_disk: bool = False, blend_special_str: str = " # # ", recomp_ratio: float = 0.15
):
    """Setup LMCache environment variables for blending."""
    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True" # enable/disable cacheblend
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    
    # Set recomputation ratio
    os.environ["LMCACHE_RECOMP_RATIO"] = str(recomp_ratio)
    logger.info(f"Setting recomputation ratio to {recomp_ratio}")

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


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    """Build LLM with LMCache integration."""
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm = LLM(
        model=model,
        kv_transfer_config=ktc, 
        max_model_len=8000,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        enforce_eager=True # disable torch compile
        # tensor_parallel_size=2 # use more than one gpu for big models
    )

    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)

def create_blended_prompt(tokenizer, contexts, question, blend_special_str):
    sys_prompt = tokenizer.encode(
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n"
    )
    
    blend_separator = tokenizer.encode(blend_special_str)[1:]
    logger.info(f"Blend separator tokens: {blend_separator}")
    
    # Encode each context as a separate chunk
    context_chunks = []
    for idx, ctx in enumerate(contexts):
        ctx_text = ctx['text']
        ctx_tokens = tokenizer.encode(ctx_text)[1:]
        context_chunks.append(ctx_tokens)
        logger.info(f"Context chunk {idx} length: {len(ctx_tokens)}")
    
    # Build the complete prompt
    prompt_tokens = sys_prompt
    
    # Add each context chunk with separators
    for ctx_chunk in context_chunks:
        prompt_tokens = prompt_tokens + blend_separator + ctx_chunk
    
    # Add the question with separator
    question_text = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. " \
                   f"Question: {normalize_question(question)}\nAnswer:"
    question_tokens = tokenizer.encode(question_text)[1:]
    logger.info(f"Question tokens length: {len(question_tokens)}")
    prompt_tokens = prompt_tokens + blend_separator + question_tokens
    
    # Log the structure of the prompt for debugging
    logger.info(f"Total prompt length: {len(prompt_tokens)}")
    logger.info(f"Number of chunks: {len(context_chunks) + 1}")  # +1 for question
    
    return (prompt_tokens)


def run_generation_with_timing(llm, prompt_tokens, sampling_params, description, request_num=None):
    """Run generation and measure timing."""
    print(f"\n--- Running {description} ---")
    print(f"Prompt length: {len(prompt_tokens)} tokens")
    
    # Get cache engine for debugging
    cache_engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    
    try:
        # Check cache status before generation
        if cache_engine is not None:
            # Use token_database to process tokens and get cache keys
            cache_keys = []
            for start, end, key in cache_engine.token_database.process_tokens(prompt_tokens):
                cache_hit = cache_engine.storage_manager.contains(key)
                cache_keys.append((key, start, end, cache_hit))
                logger.info(f"Request {request_num} - Cache key: {key}, Range: {start}-{end}, Hit: {cache_hit}")
        
        start_time = time.time()
        outputs = llm.generate(prompt_token_ids=prompt_tokens, sampling_params=sampling_params)
        end_time = time.time()
        
        generation_time = end_time - start_time
        generated_text = outputs[0].outputs[0].text
        
        print(f"Generated answer: {generated_text!r}")
        print(f"Generation time: {generation_time:.3f} seconds")
        
        # Check cache status after generation
        if cache_engine is not None:
            for key, start, end, _ in cache_keys:
                cache_hit = cache_engine.storage_manager.contains(key)
                logger.info(f"Request {request_num} - After generation - Cache key: {key}, Range: {start}-{end}, Hit: {cache_hit}")
        
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
    # random.seed(42)
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
        default=" # # ",
        help="Specify the special separators to separate chunks (default: '# #')",
    )
    parser.add_argument(
        "--dataset-path",
        default="wikimqa_s.json",
        help="Path to dataset (default: wikimqa_s.json)",
    )
    parser.add_argument(
        "--recomp-ratio",
        type=float,
        default=0.15,
        help="Recomputation ratio for blending (default: 0.15)",
    )
    parser.add_argument(
        "--debug-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the debug level (default: INFO)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set logging level
    logging.getLogger("cacheblend_debug").setLevel(getattr(logging, args.debug_level))
    
    # Configuration
    lmcache_connector = "LMCacheConnectorV1"
    model = "mistralai/Mistral-7B-Instruct-v0.2"
    # model = "mistralai/Mixtral-8x7B-Instruct-v0.1" # doesn't work
    # model = "meta-llama/Llama-3.1-8B-Instruct" # nonsense tokenizer
    # model = "meta-llama/Meta-Llama-3-8B-Instruct" # better but still kind of nonsense
    # model = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # model too big
    
    # Setup environment
    setup_environment_variables(args.use_disk, args.blend_special_str, args.recomp_ratio)
    
    # Load dataset and tokenizer
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Select a single example to demonstrate cache blending
    example = dataset[9]
    print(f"Selected example for cache blending demonstration")
    print(f"Question: {example['question']}")
    print(f"Expected answers: {example['answers'][0]}")
    print(f"Number of contexts: {len(example['ctxs'])}")
    
    # Create two different orderings of the same contexts
    original_contexts = example["ctxs"]
    reordered_contexts = create_reordered_contexts(original_contexts)
    reordered_contexts2 = create_reordered_contexts(reordered_contexts)
    # original_contexts.pop(5) # remove middle context and see effect on kvcache hit
    
    # Track results
    first_time = 0.0
    second_time = 0.0
    third_time = 0.0
    f1_scores = []
    first_answer = ""
    second_answer = ""
    third_answer = ""
    
    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=32)
        
        print(f"\n{'='*60}")
        print("FIRST REQUEST - Original context order (establishes cache)")
        print(f"{'='*60}")
        
        # Create blended prompt with original context order
        first_prompt_tokens = create_blended_prompt(
            tokenizer, 
            original_contexts, 
            example["question"], 
            args.blend_special_str
        )
        
        # Run first generation (establishes cache)
        first_answer, first_time = run_generation_with_timing(
            llm, first_prompt_tokens, sampling_params, "First request (original order)", request_num=1
        )
        
        # Calculate F1 score for first answer
        f1_first = max([compute_f1(first_answer, answer[0], tokenizer) 
                        for answer in example["answers"]])
        f1_scores.append(f1_first)
        print(f"F1 Score: {f1_first:.3f}")
        
        time.sleep(1)
        
        print(f"\n{'='*60}")
        print("SECOND REQUEST - Reordered contexts (benefits from cache blending)")
        print(f"{'='*60}")
        
        # Create blended prompt with reordered contexts
        second_prompt_tokens = create_blended_prompt(
            tokenizer, 
            reordered_contexts, 
            # original_contexts,
            "Where is the director of the movie Wine Of Morning employed?",
            # "At what company does the director of film Wine Of Morning hold a position?",
            # example["question"], 
            args.blend_special_str
        )
        
        # Run second generation (should benefit from cache blending)
        second_answer, second_time = run_generation_with_timing(
            llm, second_prompt_tokens, sampling_params, "Second request (reordered contexts)", request_num=2
        )
        
        # Calculate F1 score for second answer
        f1_second = max([compute_f1(second_answer, answer[0], tokenizer) 
                        for answer in example["answers"]])
        f1_scores.append(f1_second)
        print(f"F1 Score: {f1_second:.3f}")
        
        time.sleep(1)
        
        print(f"\n{'='*60}")
        print("THIRD REQUEST - Modified question with same contexts (testing cache reuse)")
        print(f"{'='*60}")
        
        # Create blended prompt with the same reordered contexts but a slightly different question
        third_prompt_tokens = create_blended_prompt(
            tokenizer, 
            # original_contexts, 
            reordered_contexts2,
            # example["question"], 
            "Can you tell me the workplace of Wine Of Morning's director?",
            args.blend_special_str
        )
        
        # Run third generation (should also benefit from cache blending)
        third_answer, third_time = run_generation_with_timing(
            llm, third_prompt_tokens, sampling_params, "Third request (modified question, same contexts)", request_num=3
        )
        
        # Calculate F1 score for third answer
        f1_third = max([compute_f1(third_answer, answer[0], tokenizer) 
                       for answer in example["answers"]])
        f1_scores.append(f1_third)
        print(f"F1 Score: {f1_third:.3f}")
    
    # Print summary results
    print(f"\n{'='*60}")
    print("CACHE BLENDING DEMONSTRATION RESULTS")
    print(f"{'='*60}")
    print(f"First request time (original order): {first_time:.3f} seconds")
    print(f"Second request time (reordered contexts): {second_time:.3f} seconds")
    print(f"Third request time (modified question): {third_time:.3f} seconds")
    
    if second_time < first_time:
        speedup = (first_time - second_time) / first_time * 100
        print(f"\nCACHE BLENDING SPEEDUP (second request): {speedup:.1f}% faster!")
        print(f"Time saved: {first_time - second_time:.3f} seconds")
    
    if third_time < first_time:
        speedup = (first_time - third_time) / first_time * 100
        print(f"\nCACHE BLENDING SPEEDUP (third request): {speedup:.1f}% faster!")
        print(f"Time saved: {first_time - third_time:.3f} seconds")
    
    print(f"\nF1 Scores:")
    print(f"  First request: {f1_scores[0]:.3f}")
    print(f"  Second request: {f1_scores[1]:.3f}")
    print(f"  Third request: {f1_scores[2]:.3f}")
    
    print(f"\nBlend separator used: '{args.blend_special_str}'")
    print(f"Chunk size: {os.environ.get('LMCACHE_CHUNK_SIZE', 'default')} tokens")



if __name__ == "__main__":
    main()
