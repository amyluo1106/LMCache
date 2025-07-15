# Standard
from dataclasses import asdict
import argparse
import contextlib
import os
import time

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(
    use_disk: bool = False, blend_special_str: str = " # # "
):
    # LMCache-related environment variables

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
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
    )

    llm = LLM(**asdict(llm_args))

    # llm = LLM(
    #     model=model,
    #     kv_transfer_config=ktc,
    #     max_model_len=8000,
    #     gpu_memory_utilization=0.8,
    #     enable_prefix_caching=False,
    # )

    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(prompt_token_ids=prompt, sampling_params=sampling_params)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        # import pdb; pdb.set_trace()
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser()
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

    return parser.parse_args()


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    # model = "mistralai/Mistral-7B-Instruct-v0.2"
    # model = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    setup_environment_variables(args.use_disk, args.blend_special_str)

    tokenizer = AutoTokenizer.from_pretrained(model)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        # This example script runs two requests with a shared prefix.
        # Define the shared prompt and specific prompts
        sys_prompt = tokenizer.encode("You are a very helpful assistant.")
        chunk1_prompt = tokenizer.encode("Hello, how are you?" * 500)[1:]
        chunk2_prompt = tokenizer.encode("Hello, what's up?" * 500)[1:]
        blend_special_str = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))[1:]

        # chunk1_prompt = tokenizer.encode("In the summer of 1492, Christopher Columbus, an Italian navigator, embarked on a daring voyage westward across the Atlantic Ocean. He commanded three ships: the Santa María, the Pinta, and the Niña. His primary objective was to find a westward sea route to Asia, which was known for its valuable spices and silks. However, instead of reaching Asia, he landed in the Americas, a continent previously unknown to Europeans. His first landfall was on October 12, 1492, in the Bahamas, a chain of islands in the Caribbean Sea. He named the island San Salvador, meaning 'Holy Savior.' Columbus believed he had reached the East Indies, and he referred to the native inhabitants as 'Indians'.")[1:]
        # chunk2_prompt = tokenizer.encode("This journey marked the beginning of a new era of exploration and colonization, although it also had devastating consequences for the indigenous populations of the Americas. Over the next several decades, European powers established settlements and exploited the vast resources of the continent. The exchange of goods, ideas, and diseases between Europe and the Americas, known as the Columbian Exchange, had a profound impact on both sides of the Atlantic. New crops like potatoes and maize were introduced to Europe, while horses and cattle were brought to the Americas.")[1:]
        # chunk3_prompt = tokenizer.encode("Columbus made three more voyages to the Americas between 1493 and 1504, exploring more of the Caribbean and the coasts of Central and South America. Despite his initial belief, he eventually came to understand that he had discovered a 'New World.' However, his legacy remains controversial, with many critics highlighting the negative impact of his expeditions on indigenous cultures and populations")[1:]

        first_prompt = (
            sys_prompt
            + blend_special_str
            + chunk1_prompt
            + blend_special_str
            + chunk2_prompt
            + blend_special_str
            + tokenizer.encode("Hello, my name is")[1:]
        )

        second_prompt = (
            sys_prompt
            + blend_special_str
            + chunk2_prompt
            + blend_special_str
            + chunk1_prompt
            + blend_special_str
            # + tokenizer.encode("Hello, my name is")[1:]
            + tokenizer.encode("Hello, how are you?")[1:]
        )

        # sys_prompt = tokenizer.encode("Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n")

        # second_prompt = (
        #     sys_prompt
        #     + blend_special_str
        #     + chunk1_prompt
        #     + blend_special_str
        #     + chunk2_prompt
        #     + blend_special_str
        #     + chunk3_prompt
        #     + blend_special_str
        #     + tokenizer.encode("On which specific island did Columbus first make landfall in the Americas?")[1:]
        # )

        # first_prompt = (
        #     sys_prompt
        #     + blend_special_str
        #     + chunk2_prompt
        #     + blend_special_str
        #     + chunk1_prompt
        #     + blend_special_str
        #     + chunk3_prompt
        #     + blend_special_str
        #     + tokenizer.encode("What does the passage suggest about the long-term consequences of Columbus's voyages, beyond the immediate discoveries?")[1:]
        # )

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=500)

        # Print the first output
        print_output(llm, first_prompt, sampling_params, "first")

        time.sleep(1)

        # print the second output
        print_output(llm, second_prompt, sampling_params, "second")

        # print_output(llm, first_prompt, sampling_params, "third")


if __name__ == "__main__":
    main()
