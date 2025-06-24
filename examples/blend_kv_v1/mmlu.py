import numpy as np
import random
from datasets import load_dataset
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from transformers import AutoTokenizer
import time

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
import pandas as pd
import re
import contextlib


# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define special tokens for blend separation
BLEND_SEPARATOR = "# #"

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
        # kv_transfer_config=ktc,  # CacheBlend
        max_model_len=8000,
        gpu_memory_utilization=0.7,
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
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
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

# Select 2 subjects
num_subjects_to_select = 2
# selected_subjects = random.sample(all_subjects, num_subjects_to_select)
selected_subjects = ["philosophy", "history"]

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

def evaluate_llama_examples(llm, tokenizer, subject_examples, num_examples=200, show_examples=5):
    """Evaluate using Llama's pre-defined prompts with blend separators and tokenized inputs."""
    
    total = min(num_examples, len(subject_examples))
    indices = random.sample(range(len(subject_examples)), total)
    
    print(f"Evaluating {total} examples from Llama's MMLU dataset...")
    
    # Lists to store data
    prompt_texts = []  # For display purposes
    prompt_token_ids_list = []  # For feeding to the model
    correct_answers = []
    questions = []
    choices_list = []
    num_shots_list = []
    
    # Timing variables
    preparation_start = time.time()
    
    # Get token IDs for the blend separator
    blend_separator_tokens = tokenizer.encode(BLEND_SEPARATOR, add_special_tokens=False)
    
    for idx in tqdm(indices, desc="Preparing prompts"):
        example = subject_examples[idx]
        
        # Extract information from the example
        input_question = example["input_question"]
        
        # Get the correct answer
        correct_answer = example["input_correct_responses"][0] if example["input_correct_responses"] else None
        # answer_match = re.search(r'Answer:\s*([A-D])', correct_answer)
        answer_match = re.search(r'Answer:\s*([A-J])', correct_answer)
        if answer_match:
            correct_answer = answer_match.group(1)

        # Extract choices
        choices = {}
        if isinstance(example["input_choice_list"], dict):
            choices = example["input_choice_list"]
        
        # Get number of shots (if available)
        num_shots = example["eval_config"]["num_few_shot"] if "num_few_shot" in example["eval_config"] else 0
        
        # For display purposes
        raw_prompt_text = ""
        
        # This will store our final token IDs
        final_prompt_tokens = []
        
        # Get few-shot examples - assuming the first element of input_final_prompt array has these examples
        if "input_final_prompts" in example and isinstance(example["input_final_prompts"], list) and len(example["input_final_prompts"]) > 0:
            few_shot_text = example["input_final_prompts"][0]
            
            # Split to extract the few-shot examples without the current question
            parts = few_shot_text.split("\n\n")
            intro = parts[0]  # "The following are multiple choice questions..."
            raw_prompt_text += intro
            
            # Add intro tokens (but don't add special tokens like BOS)
            intro_tokens = tokenizer.encode(intro, add_special_tokens=False)
            final_prompt_tokens.extend(intro_tokens)
            
            # Process each example separately with blend separators
            if len(parts) > 2:
                examples_parts = parts[1:-1]  # Skip intro and last part (test question)
                
                # Process each example
                for i, example_part in enumerate(examples_parts):
                    # Add blend separator before each example (except the first one)
                    if i > 0 or final_prompt_tokens:  # If not the first item or if we already have tokens
                        final_prompt_tokens.extend(blend_separator_tokens)
                        raw_prompt_text += BLEND_SEPARATOR
                    
                    # Add example tokens (no special tokens)
                    example_tokens = tokenizer.encode(example_part, add_special_tokens=False)
                    final_prompt_tokens.extend(example_tokens)
                    raw_prompt_text += example_part
            
            # Format the test question with choices
            question_text = f"Question: {input_question}\n\nChoices:\n"
            for choice_key in sorted(choices.keys()):
                question_text += f"{choice_key}. {choices[choice_key]}\n"
            question_text += "Provide the letter corresponding to the correct answer."
            
            # Add blend separator before the test question
            if final_prompt_tokens:
                final_prompt_tokens.extend(blend_separator_tokens)
                raw_prompt_text += BLEND_SEPARATOR
            
            # Add test question tokens (no special tokens)
            test_question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
            final_prompt_tokens.extend(test_question_tokens)
            raw_prompt_text += question_text
            
        else:
            # Fallback if we can't extract few-shot examples
            question_text = f"Answer the following multiple choice question. Select the letter of the correct answer. Do not repeat the question\n\nQuestion: {input_question}\n\nChoices:\n"
            for choice_key in sorted(choices.keys()):
                question_text += f"{choice_key}. {choices[choice_key]}\n"
            question_text += "Answer:"
            
            # Tokenize directly (no special tokens)
            question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
            final_prompt_tokens.extend(question_tokens)
            raw_prompt_text = question_text
        
        # Add BOS token at the beginning if needed
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is not None:
            final_prompt_tokens = [bos_token_id] + final_prompt_tokens
        
        prompt_texts.append(raw_prompt_text)  # Keep text for display
        prompt_token_ids_list.append(final_prompt_tokens)  # Store token IDs for model input
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

    
    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.0, max_tokens=500)
    
    # Generate outputs using token IDs directly
    print("Generating answers...")
    
    # Start generation timing
    generation_start = time.time()
    
    # Run generation
    outputs = llm.generate(prompt_token_ids=prompt_token_ids_list, sampling_params=sampling_params)
    
    # End generation timing
    generation_end = time.time()
    generation_time = generation_end - generation_start
    
    print(f"LLM generation time: {generation_time:.2f} seconds ({generation_time/total:.4f} sec/example)")
    print(f"Generation throughput: {total_tokens/generation_time:.1f} tokens/sec")
    
    # Calculate total generated tokens
    total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Output generation throughput: {total_output_tokens/generation_time:.1f} tokens/sec")
    
    # Start parsing timing
    parsing_start = time.time()
    
    # Calculate accuracy
    correct = 0
    for i, (output, correct_answer) in enumerate(zip(outputs, correct_answers)):
        response = output.outputs[0].text.strip()
        
        # Parse the letter from the response
        parsed_letter = None
        
        # Try different parsing approaches
        # 1. Look for "Answer: X" pattern
        # answer_match = re.search(r'Answer:\s*([A-D])', response)
        # if answer_match:
        #     parsed_letter = answer_match.group(1)
        
        # # 2. Look for standalone A, B, C, or D
        # elif re.search(r'(?:^|\s|\n|[.,;:])([A-D])(?:$|\s|\n|[.,;:])', response):
        #     letter_match = re.search(r'(?:^|\s|\n|[.,;:])([A-D])(?:$|\s|\n|[.,;:])', response)
        #     parsed_letter = letter_match.group(1)
        
        # # 3. Just take the first letter that is A, B, C, or D in the response
        # else:
        #     for char in response:
        #         if char in "ABCD":
        #             parsed_letter = char
        #             break
        answer_match = re.search(r'Answer:\s*([A-J])', response)
        if answer_match:
            parsed_letter = answer_match.group(1)
        
        # 2. Look for standalone A, B, C, or D
        elif re.search(r'(?:^|\s|\n|[.,;:])([A-J])(?:$|\s|\n|[.,;:])', response):
            letter_match = re.search(r'(?:^|\s|\n|[.,;:])([A-J])(?:$|\s|\n|[.,;:])', response)
            parsed_letter = letter_match.group(1)
        
        # 3. Just take the first letter that is A, B, C, or D in the response
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
                    print(f"{choice_key}. {choices[choice_key]}")
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
        "total_output_tokens": total_output_tokens,
        "generation_throughput": total_tokens/generation_time if generation_time > 0 else 0,
        "output_throughput": total_output_tokens/generation_time if generation_time > 0 else 0
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
        print(f"  - Output throughput: {timing['output_throughput']:.1f} tokens/sec")

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
