"""
Run script with:

PYTHONPATH=$(pwd) /opt/nvidia/nsight-systems/2025.1.3/bin/nsys profile \
    --trace=cuda,nvtx --trace-fork-before-exec=true --capture-range=cudaProfilerApi \
    --cuda-graph-trace=node --sample=none --cpuctxsw=none \
    python profile_cacheblend.py 
"""

# add nvtx range push to check attention token recompute (in lmcache/v1/compute/blend/blender.py)

# Standard
import argparse
import contextlib
import os
import time
import logging
import random
import torch

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cacheblend_profiler")

def setup_environment_variables(
    use_disk: bool = False, blend_special_str: str = " # # ", recomp_ratio: float = 0.15
):
    """Setup LMCache environment variables for blending."""
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"  # enable/disable cacheblend
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    
    # Set recomputation ratio
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIO"] = str(recomp_ratio)
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
def build_llm_with_lmcache(lmcache_connector: str, model: str, tensor_parallel_size: int = 1):
    """Build LLM with LMCache integration."""
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm = LLM(
        model=model,
        kv_transfer_config=ktc, 
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        enforce_eager=True,  # disable torch compile
        tensor_parallel_size=tensor_parallel_size  # use more than one gpu for big models
    )

    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def create_blended_prompt(tokenizer, chunks, question, blend_special_str):
    """Create a blended prompt with chunks and a question."""
    sys_prompt = tokenizer.encode(
        "Answer the question based on the given passages. Be concise and accurate.\n\nThe following are given passages.\n"
    )
    
    blend_separator = tokenizer.encode(blend_special_str)[1:]
    logger.info(f"Blend separator tokens: {blend_separator}")
    
    # Encode each chunk
    chunk_tokens = []
    for idx, chunk in enumerate(chunks):
        tokens = tokenizer.encode(chunk)[1:]
        chunk_tokens.append(tokens)
        logger.info(f"Chunk {idx} length: {len(tokens)}")
    
    # Build the complete prompt
    prompt_tokens = sys_prompt
    
    # Add each chunk with separators
    for chunk in chunk_tokens:
        prompt_tokens = prompt_tokens + blend_separator + chunk
    
    # Add the question with separator
    question_text = f"\n\nQuestion: {question}\nAnswer:"
    question_tokens = tokenizer.encode(question_text)[1:]
    logger.info(f"Question tokens length: {len(question_tokens)}")
    prompt_tokens = prompt_tokens + blend_separator + question_tokens
    
    # Log the structure of the prompt for debugging
    logger.info(f"Total prompt length: {len(prompt_tokens)}")
    logger.info(f"Number of chunks: {len(chunk_tokens) + 1}")  # +1 for question
    
    return prompt_tokens


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
            hit_count = 0
            total_count = 0
            for key, start, end, _ in cache_keys:
                cache_hit = cache_engine.storage_manager.contains(key)
                logger.info(f"Request {request_num} - After generation - Cache key: {key}, Range: {start}-{end}, Hit: {cache_hit}")
                total_count += 1
                if cache_hit:
                    hit_count += 1
            
            if total_count > 0:
                hit_rate = hit_count / total_count * 100
                print(f"Cache hit rate: {hit_rate:.2f}% ({hit_count}/{total_count})")
        
        return generated_text, generation_time
        
    except Exception as e:
        print(f"Error during generation: {e}")
        print(f"Prompt token count: {len(prompt_tokens)}")
        raise


def create_test_chunks():
    """Create test chunks for profiling (targeting 3000-4000 tokens total)."""
    chunks = [
        # Chunk 1: Eiffel Tower
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
        "Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized "
        "by some of France's leading artists and intellectuals for its design, but it has become a global "
        "cultural icon of France and one of the most recognizable structures in the world. "
        "The Eiffel Tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. "
        "Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument "
        "to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building "
        "in New York City was finished in 1930. It was the first structure in the world to surpass both the 200-metre and 300-metre mark in height. "
        "Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). "
        "Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct. "
        "The tower has three levels for visitors, with restaurants on the first and second levels. The top level's upper platform is 276 m (906 ft) "
        "above the ground – the highest observation deck accessible to the public in the European Union. Tickets can be purchased to ascend by stairs or lift to the first and second levels. "
        "The climb from ground level to the first level is over 300 steps, as is the climb from the first level to the second, making a total of over 600 steps. "
        "The tower has become the most-visited paid monument in the world; 6.91 million people ascended it in 2015. The tower received its 250 millionth visitor in 2010. "
        "The tower is a featured element in the climax of the 1951 film The Day the Earth Stood Still, the 1985 James Bond film A View to a Kill, and was also featured in "
        "The Lavender Hill Mob, Casino Royale, Amélie, and the 1987 film The Aristocats.",
        
        # Chunk 2: Amazon Rainforest
        "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon "
        "biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 "
        "(2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. "
        "This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, "
        "with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, "
        "Ecuador, Bolivia, Guyana, Suriname, and French Guiana. Four nations have 'Amazonas' as the name of one of their first-level administrative regions, "
        "and France uses the name 'Guiana Amazonian Park' for its rainforest protected area. The Amazon represents over half of the planet's remaining rainforests, "
        "and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species. "
        "More than 30 million people of 350 different ethnic groups live in the Amazon, which are subdivided into 9 different national political systems and 3,344 formally acknowledged indigenous territories. "
        "Indigenous peoples make up 9% of the total population, with 60 of the groups remaining largely isolated. The rainforest likely formed during the Eocene era (from 56 million years to 33.9 million years ago). "
        "It appeared following a global reduction of tropical temperatures when the Atlantic Ocean had widened sufficiently to provide a warm, moist climate to the Amazon basin. "
        "The rainforest has been in existence for at least 55 million years, and most of the region remained free of savanna-type biomes at least until the current ice age when the climate was drier and savanna more widespread. "
        "Following the Cretaceous–Paleogene extinction event, the extinction of the dinosaurs and the wetter climate may have allowed the tropical rainforest to spread out across the continent. "
        "From 66 to 34 Mya, the rainforest extended as far south as 45°. Climate fluctuations during the last 34 million years have allowed savanna regions to expand into the tropics. "
        "During the Oligocene, for example, the rainforest spanned a relatively narrow band. It expanded again during the Middle Miocene, then retracted to a mostly inland formation at the last glacial maximum. "
        "However, the rainforest still managed to thrive during these glacial periods, allowing for the survival and evolution of a broad diversity of species.",
        
        # Chunk 3: Artificial Intelligence
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural "
        "intelligence displayed by humans or animals. Leading AI textbooks define the field as the study of "
        "'intelligent agents': any system that perceives its environment and takes actions that maximize its "
        "chance of achieving its goals. Some popular accounts use the term 'artificial intelligence' to describe machines "
        "that mimic 'cognitive' functions that humans associate with the human mind, such as 'learning' and 'problem solving', "
        "however, this definition is rejected by major AI researchers. AI applications include advanced web search engines, "
        "recommendation systems (used by YouTube, Amazon and Netflix), understanding human speech (such as Siri and Alexa), "
        "self-driving cars (e.g., Waymo), automated decision-making and competing at the highest level in strategic game systems "
        "(such as chess and Go). As machines become increasingly capable, tasks considered to require 'intelligence' are often "
        "removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is "
        "frequently excluded from things considered to be AI, having become a routine technology. "
        "Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves "
        "of optimism, followed by disappointment and the loss of funding (known as an 'AI winter'), followed by new approaches, success and renewed funding. "
        "AI research has tried and discarded many different approaches during its lifetime, including simulating the brain, modeling human problem solving, "
        "formal logic, large databases of knowledge and imitating animal behavior. In the first decades of the 21st century, highly mathematical statistical "
        "machine learning has dominated the field, and this technique has proved highly successful, helping to solve many challenging problems throughout industry and academia. "
        "The various sub-fields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include reasoning, "
        "knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects. General intelligence (the ability to solve "
        "an arbitrary problem) is among the field's long-term goals. To solve these problems, AI researchers have adapted and integrated a wide range of problem-solving techniques – including "
        "search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, probability and economics. AI also draws upon computer science, psychology, "
        "linguistics, philosophy, and many other fields.",
        
        # Chunk 4: Great Barrier Reef 
        "The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs "
        "and 900 islands stretching for over 2,300 kilometres over an area of approximately 344,400 square kilometres. "
        "The reef is located in the Coral Sea, off the coast of Queensland, Australia. The Great Barrier Reef can be seen from outer space "
        "and is the world's biggest single structure made by living organisms. This reef structure is composed of and built by billions of tiny organisms, "
        "known as coral polyps. It supports a wide diversity of life and was selected as a World Heritage Site in 1981. CNN labelled it one of the seven natural wonders of the world. "
        "The Queensland National Trust named it a state icon of Queensland. A large part of the reef is protected by the Great Barrier Reef Marine Park, which helps to limit the impact of "
        "human use, such as fishing and tourism. Other environmental pressures on the reef and its ecosystem include runoff, climate change accompanied by mass coral bleaching, "
        "dumping of dredging sludge and cyclic population outbreaks of the crown-of-thorns starfish. According to a study published in October 2012 by the Proceedings of the National Academy of Sciences, "
        "the reef has lost more than half its coral cover since 1985. The Great Barrier Reef has long been known to and used by the Aboriginal Australian and Torres Strait Islander peoples, "
        "and is an important part of local groups' cultures and spirituality. The reef is a very popular destination for tourists, especially in the Whitsunday Islands and Cairns regions. "
        "Tourism is an important economic activity for the region, generating over AUD$3 billion per year. In November 2014, Google launched Google Underwater Street View in 3D of the Great Barrier Reef. "
        "A March 2016 report stated that coral bleaching was more widespread than previously thought, seriously affecting the northern parts of the reef as a result of warming ocean temperatures. "
        "In April 2016, an aerial survey of the northern part of the reef found that 95% of the reefs were severely bleached, surpassing the 2015 bleaching event. As of March 2017, the Great Barrier Reef has experienced "
        "two consecutive years of severe coral bleaching, with areas that were previously untouched by high sea temperatures now experiencing extreme levels of coral decline. In April 2018, it was reported that 30% of the "
        "coral had died in the heatwaves of 2016 and 2017. In 2020, a study found that the Great Barrier Reef has lost more than half of its corals since 1995 due to warmer seas driven by climate change.",
        
        # Chunk 5: The Moon
        "The Moon is Earth's only natural satellite. At about one-quarter the diameter of Earth, it is the largest "
        "natural satellite in the Solar System relative to the size of its planet, the fifth largest satellite in "
        "the Solar System overall, and is larger than any dwarf planet. Orbiting Earth at an average distance of 384,400 km (238,900 mi), "
        "or about 30 times Earth's diameter, its gravitational influence slightly lengthens Earth's day and is the main driver of Earth's tides. "
        "The Moon is classified as a planetary-mass object and a differentiated rocky body, and lacks any significant atmosphere, hydrosphere, or magnetic field. "
        "Its surface gravity is about one-sixth of Earth's (0.1654 g); Jupiter's moon Io is the only satellite in the Solar System known to have a higher surface gravity and density. "
        "The Moon's orbit around Earth has a sidereal period of 27.3 days. During each synodic period of 29.5 days, the amount of visible surface illuminated by the Sun varies from none up to 100%, "
        "resulting in lunar phases that form the basis for the months of a lunar calendar. The Moon is tidally locked to Earth, which means that the length of a full rotation of the Moon on its own axis "
        "causes its same side (the near side) to always face Earth, and the somewhat longer lunar day is the same as the synodic period. That said, 59% of the total lunar surface can be seen from Earth "
        "through shifts in perspective due to libration. The most widely accepted origin explanation posits that the Moon formed about 4.51 billion years ago, not long after Earth, out of the debris from a "
        "giant impact between the planet and a hypothesized Mars-sized body called Theia. It then receded to a wider orbit because of tidal interaction with the Earth. The near side of the Moon is marked by "
        "dark volcanic maria ('seas'), which fill the spaces between bright ancient crustal highlands and prominent impact craters. Most of the large impact basins and mare surfaces were in place by the end "
        "of the Imbrian period, some three billion years ago. The lunar surface is relatively non-reflective, with a reflectance just slightly brighter than that of worn asphalt. However, because it has a large "
        "angular diameter, the full moon is the brightest celestial object in the night sky. The Moon's apparent size is nearly the same as that of the Sun, allowing it to cover the Sun almost completely during a "
        "total solar eclipse. This matching of apparent visual size will not continue in the far future because the Moon's distance from Earth is gradually increasing. The Soviet Union's Luna programme was the first "
        "to reach the Moon with uncrewed spacecraft in 1959; the United States' NASA Apollo program achieved the only crewed lunar missions to date, beginning with the first crewed orbital mission by Apollo 8 in 1968, "
        "and six crewed landings between 1969 and 1972, with the first being Apollo 11 in July 1969. These and later uncrewed missions returned lunar rocks that have been used to develop a detailed geological understanding "
        "of the Moon's origins, internal structure, and subsequent history. The Moon has been the subject of many works of art and literature and is a source for inspiration and research for humans for millennia."
    ]
    return chunks


def run_profiling(args):
    """Run profiling of cacheblend with different chunk orderings."""
    # Configuration
    lmcache_connector = "LMCacheConnectorV1"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Setup environment
    setup_environment_variables(args.use_disk, args.blend_special_str, args.recomp_ratio)
    
    # Load tokenizer
    print(f"Loading tokenizer for model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Create test chunks and questions
    chunks = create_test_chunks()
    question = "What is the Eiffel Tower and where is it located?"
    
    # Create different orderings of the chunks
    original_chunks = chunks
    reordered_chunks = chunks.copy()
    random.shuffle(reordered_chunks)
    
    # Different question but same topic
    alt_question = "Who designed the Eiffel Tower and when was it built?"
    
    # Track results
    first_time = 0.0
    second_time = 0.0
    third_time = 0.0
    
    with build_llm_with_lmcache(lmcache_connector, model, args.tensor_parallel_size) as llm:
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=64)
        
        # Warmup run (not profiled)    
        print("\n=== STARTING PROFILING ===")
        
        print(f"\n{'='*60}")
        print("FIRST REQUEST - Original chunk order (establishes cache)")
        print(f"{'='*60}")
        
        # Create blended prompt with original chunk order
        first_prompt_tokens = create_blended_prompt(
            tokenizer, 
            original_chunks, 
            question, 
            args.blend_special_str
        )
        
        _, first_time = run_generation_with_timing(
            llm, first_prompt_tokens, sampling_params, "First request (original order)", request_num=1
        )
        
        time.sleep(1)
        
        print(f"\n{'='*60}")
        print("SECOND REQUEST - Reordered chunks (benefits from cache blending)")
        print(f"{'='*60}")
        
        # Create blended prompt with reordered chunks
        second_prompt_tokens = create_blended_prompt(
            tokenizer, 
            reordered_chunks, 
            question, 
            args.blend_special_str
        )
        
        _, second_time = run_generation_with_timing(
            llm, second_prompt_tokens, sampling_params, "Second request (reordered chunks)", request_num=2
        )
        
        time.sleep(1)
        
        print(f"\n{'='*60}")
        print("THIRD REQUEST - Different question with original chunks")
        print(f"{'='*60}")
        
        # Create blended prompt with original chunks but different question
        third_prompt_tokens = create_blended_prompt(
            tokenizer, 
            original_chunks, 
            alt_question, 
            args.blend_special_str
        )
        
        # Run third generation with profiling
        torch.cuda.cudart().cudaProfilerStart()
        _, third_time = run_generation_with_timing(
            llm, third_prompt_tokens, sampling_params, "Third request (different question)", request_num=3
        )
        torch.cuda.cudart().cudaProfilerStop()
    
    # Print summary results
    print(f"\n{'='*60}")
    print("CACHE BLENDING PROFILING RESULTS")
    print(f"{'='*60}")
    print(f"First request time (original chunks): {first_time:.3f} seconds")
    print(f"Second request time (reordered chunks): {second_time:.3f} seconds")
    print(f"Third request time (different question): {third_time:.3f} seconds")
    
    if second_time < first_time:
        speedup = (first_time - second_time) / first_time * 100
        print(f"\nCACHE BLENDING SPEEDUP (second request): {speedup:.1f}% faster!")
        print(f"Time saved: {first_time - second_time:.3f} seconds")
    
    if third_time < first_time:
        speedup = (first_time - third_time) / first_time * 100
        print(f"\nCACHE BLENDING SPEEDUP (third request): {speedup:.1f}% faster!")
        print(f"Time saved: {first_time - third_time:.3f} seconds")
    
    print(f"\nBlend separator used: '{args.blend_special_str}'")
    print(f"Recomputation ratio: {args.recomp_ratio}")
    print(f"Chunk size: {os.environ.get('LMCACHE_CHUNK_SIZE', 'default')} tokens")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile cache blending performance")
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
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for model loading (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set logging level
    logging.getLogger("cacheblend_profiler").setLevel(getattr(logging, args.debug_level))
    
    print("=== CACHEBLEND PROFILING ===")
    print(f"Model: meta-llama/Llama-3.1-8B-Instruct")
    print(f"Recomputation ratio: {args.recomp_ratio}")
    print(f"Blend special string: '{args.blend_special_str}'")
    print(f"Using disk backend: {args.use_disk}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    
    run_profiling(args)


if __name__ == "__main__":
    main()
