import random
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from transformers import AutoTokenizer
import time

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
import pandas as pd
import re
import contextlib
import os

import logging

# Configure the LMCache logger to only show errors or higher
logging.getLogger("lmcache").setLevel(logging.ERROR)

# Define special tokens for blend separation
BLEND_SEPARATOR = " # # "

os.environ["LMCACHE_CHUNK_SIZE"] = "256"

os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
os.environ["LMCACHE_BLEND_SPECIAL_STR"] = BLEND_SEPARATOR
os.environ["LMCACHE_USE_LAYERWISE"] = "True"

os.environ["LMCACHE_BLEND_RECOMPUTE_RATIO"] = str(0.15) 

os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "40"

@contextlib.contextmanager
def build_llm_with_lmcache(model_name, lmcache_connector):
    """Build LLM with LMCache integration."""
    print(f"Initializing LLM with LMCache ({lmcache_connector})...")
    
    # Configure KV transfer for cache blending
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    # Initialize the model with LMCache
    llm = LLM(
        model=model_name,
        kv_transfer_config=ktc,  
        max_model_len=8000,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        enforce_eager=True,  # disable torch compile
        tensor_parallel_size=4  # adjust based on available GPUs
    )

    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME) 

# Initialize model and load dataset
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
lmcache_connector = "LMCacheConnectorV1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load MMLU evaluation details from Llama-3.1-8B-evals
print("Loading Llama 3.1 MMLU evaluation data...")
llama_mmlu_data = load_dataset(
    "meta-llama/Llama-3.1-8B-Instruct-evals",
    name="Llama-3.1-8B-Instruct-evals__mmlu_pro__details",
    split="latest"
)

# Get all unique subject names from the dataset
all_subjects = list(set(llama_mmlu_data["subtask_name"]))
print(f"Found {len(all_subjects)} unique subjects in the MMLU dataset")

selected_subjects = ["mmlu_pro_chat.economics"]
# selected_subjects = ['mmlu_pro_chat.math', 'mmlu_pro_chat.economics', 'mmlu_pro_chat.health', 'mmlu_pro_chat.other', 'mmlu_pro_chat.computer_science', 'mmlu_pro_chat.psychology', 'mmlu_pro_chat.engineering', 'mmlu_pro_chat.business', 'mmlu_pro_chat.philosophy', 'mmlu_pro_chat.chemistry', 'mmlu_pro_chat.law', 'mmlu_pro_chat.physics', 'mmlu_pro_chat.history', 'mmlu_pro_chat.biology']

print(f"\nSelected {len(selected_subjects)} subjects for evaluation:")
for subject in selected_subjects:
    print(f"  - {subject}")

# Group examples by subtask
examples_by_subject = {}
for subject in selected_subjects:
    subject_examples = llama_mmlu_data.filter(
        lambda example: example["subtask_name"] == subject
    )
    examples_by_subject[subject] = subject_examples
    print(f"  - {subject}: {len(subject_examples)} examples")


def evaluate_llama_examples(llm, tokenizer, subject_examples, num_examples=15, show_examples=5, randomize_shots=True):
    """Evaluate using Llama's pre-defined prompts with blend separators and tokenized inputs."""
    
    total = min(num_examples, len(subject_examples))
    random.seed(42)
    indices = random.sample(range(total), total)
    
    print(f"Evaluating {total} examples from Llama's MMLU dataset...")
    
    # Lists to store data
    prompt_token_ids_list = []
    correct_answers = []
    questions = []
    choices_list = []
    num_shots_list = []
    
    # Timing variables
    preparation_start = time.time()
    
    # Get token IDs for the blend separator
    blend_separator_tokens = tokenizer.encode(BLEND_SEPARATOR)[1:]
    
    for idx in tqdm(indices, desc="Preparing prompts"):
        example = subject_examples[idx]
        
        # Extract information from the example
        input_question = example["input_question"]
        
        # Get the correct answer
        correct_answer = example["input_correct_responses"][0] if example["input_correct_responses"] else None
        answer_match = re.search(r'Answer:\s*([A-J])', correct_answer)
        if answer_match:
            correct_answer = answer_match.group(1)

        # Extract choices
        choices = {}
        if isinstance(example["input_choice_list"], dict):
            choices = example["input_choice_list"]
        
        # Get number of shots
        num_shots = example["eval_config"]["num_few_shot"] if "num_few_shot" in example["eval_config"] else 0
        
        # Get the raw prompt text
        few_shot_text = example["input_final_prompts"][0]
        
        # Split to extract parts
        examples = few_shot_text.split("<|start_header_id|>user<|end_header_id|>")

        for i in range(len(examples)):
            examples[i] = "<|start_header_id|>user<|end_header_id|>" + examples[i]

        examples.pop(0)
        
        # The last example in our list should be the final test question
        if examples and int(num_shots) > 0:
            # Take the last example as our final test question
            final_test_question = examples.pop()  # Remove and store the last example
        
        # Now we can shuffle just the examples (not the final test question)
        if randomize_shots and examples:
            # random.seed(42)
            random.shuffle(examples)

        context_chunks = []
        for idx, ctx in enumerate(examples):
            ctx_tokens = tokenizer.encode(ctx)[1:]
            context_chunks.append(ctx_tokens)

        final_prompt_tokens = tokenizer.encode("Choose the best multiple choice answer.\n")

        for ctx_chunk in context_chunks:
            final_prompt_tokens = final_prompt_tokens + blend_separator_tokens + ctx_chun

        question_tokens = tokenizer.encode(final_test_question)[1:]
        final_prompt_tokens = final_prompt_tokens + blend_separator_tokens + question_tokens
        
        prompt_token_ids_list.append((final_prompt_tokens))
        correct_answers.append(correct_answer.strip('"'))
        questions.append(input_question)
        choices_list.append(choices)
        num_shots_list.append(num_shots)
    
    # Preparation time
    preparation_end = time.time()
    preparation_time = preparation_end - preparation_start
    print(f"Prompt preparation time: {preparation_time:.2f} seconds ({preparation_time/total:.4f} sec/example)")
    
    # Calculate average and total token counts
    total_tokens = sum(len(tokens) for tokens in prompt_token_ids_list)
    avg_tokens = total_tokens / len(prompt_token_ids_list) if prompt_token_ids_list else 0
    print(f"Total tokens across all prompts: {total_tokens}")

    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=2000) 
    
    print("Generating answers...")
    
    # Start generation timing
    generation_start = time.time()
    
    # Run generation
    outputs = []
    for p in prompt_token_ids_list:
        out = llm.generate(prompt_token_ids=p, sampling_params=sampling_params)
        outputs.append(out[0].outputs[0].text)
        time.sleep(1)
    
    # End generation timing
    generation_end = time.time()
    generation_time = generation_end - generation_start
    
    print(f"LLM generation time: {generation_time:.2f} seconds ({generation_time/total:.4f} sec/example)")
    print(f"Generation throughput: {total_tokens/generation_time:.1f} tokens/sec")
    
    parsing_start = time.time()
    
    # Calculate accuracy
    correct = 0
    for i, (output, correct_answer) in enumerate(zip(outputs, correct_answers)):
        response = output.strip()
        
        # Parse the letter from the response
        parsed_letter = None
        
        # Try different parsing approaches
        answer_match = re.search(r'Answer:\s*([A-J])', response)
        if answer_match:
            parsed_letter = answer_match.group(1)
        
        # Look for "The best answer is X" pattern
        elif re.search(r'best answer is ([A-J])', response, re.IGNORECASE):
            best_match = re.search(r'best answer is ([A-J])', response, re.IGNORECASE)
            parsed_letter = best_match.group(1)
        
        # Look for standalone A-J
        elif re.search(r'(?:^|\s|\n|[.,;:])([A-J])(?:$|\s|\n|[.,;:])', response):
            letter_match = re.search(r'(?:^|\s|\n|[.,;:])([A-J])(?:$|\s|\n|[.,;:])', response)
            parsed_letter = letter_match.group(1)
        
        # Just take the first letter that is A-J in the response
        else:
            for char in response:
                if char in "ABCDEFGHIJ":
                    parsed_letter = char
                    break
        
        # Check if parsed letter matches the correct answer
        is_correct = parsed_letter == correct_answer
        if is_correct:
            correct += 1
        
        # Print examples for debugging
        if i < show_examples:
            print("\n" + "="*50)
            print(f"EXAMPLE {i+1} ({num_shots_list[i]}-shot, {len(prompt_token_ids_list[i])} tokens):")
            print(f"QUESTION: {questions[i]}")
            if choices_list[i]:
                print("CHOICES:")
                for choice_key in sorted(choices_list[i].keys()):
                    print(f"{choice_key}. {choices_list[i][choice_key]}")
            print(f"CORRECT ANSWER: {correct_answer}")
            print(f"MODEL RESPONSE: {response}")
            print(f"PARSED LETTER: {parsed_letter}")
            print(f"RESULT: {'✓ Correct' if is_correct else '✗ Incorrect'}")
            print("="*50)
    
    # End parsing timing
    parsing_end = time.time()
    parsing_time = parsing_end - parsing_start
    print(f"Response parsing time: {parsing_time:.2f} seconds ({parsing_time/total:.4f} sec/example)")
    
    # Calculate total time
    total_time = preparation_time + generation_time + parsing_time
    print(f"Total evaluation time: {total_time:.2f} seconds ({total_time/total:.4f} sec/example)")
    
    accuracy = correct / total
    print(f"Evaluation complete. Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    timing_stats = {
        "preparation_time": preparation_time,
        "generation_time": generation_time,
        "parsing_time": parsing_time,
        "total_time": total_time,
        "tokens_per_prompt": avg_tokens,
        "total_tokens": total_tokens,
        "generation_throughput": total_tokens/generation_time if generation_time > 0 else 0,
    }
    
    return accuracy, timing_stats

# Use the LMCache-enabled LLM for evaluation
with build_llm_with_lmcache(model_name, lmcache_connector) as llm:
    # Evaluate each subject
    results = {}
    timing_results = {}
    
    for subject, examples in examples_by_subject.items():
        print(f"\n{'='*70}")
        print(f"EVALUATING: {subject.upper()}")
        print(f"{'='*70}")

        # Evaluate using Llama's examples with tokenizer and blend separators
        accuracy, timing_stats = evaluate_llama_examples(llm, tokenizer, examples)
        
        # Store the results
        results[subject] = accuracy
        timing_results[subject] = timing_stats

    # Summarize results
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    for subject, accuracy in results.items():
        timing = timing_results[subject]
        print(f"\n{subject.upper()}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Timing statistics:")
        print(f"  - Preparation: {timing['preparation_time']:.2f}s")
        print(f"  - LLM Generation: {timing['generation_time']:.2f}s")
        print(f"  - Response Parsing: {timing['parsing_time']:.2f}s")
        print(f"  - Total Time: {timing['total_time']:.2f}s")
        print(f"  - Generation throughput: {timing['generation_throughput']:.1f} tokens/sec")

# Additional information about the dataset
print("\n" + "="*70)
print("DATASET INFORMATION")
print("="*70)

# Display distribution of number of shots in the dataset
shot_counts = {}
for subject in selected_subjects:
    examples = examples_by_subject[subject]
    shots = [ex["eval_config"]["num_few_shot"] if "num_few_shot" in ex["eval_config"] else 0 for ex in examples]
    shot_counts[subject] = dict(pd.Series(shots).value_counts().items())

for subject, counts in shot_counts.items():
    print(f"\n{subject.upper()}:")
    for shots, count in sorted(counts.items()):
        print(f"  {shots}-shot examples: {count}")
