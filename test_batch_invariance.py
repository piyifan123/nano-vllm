import os
import time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup model and tokenizer
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # Use temperature=0 for greedy and deterministic output
    sampling_params = SamplingParams(temperature=0, max_tokens=32)

    prompt = "Generate 100 random numbers. Go directly into it, don't say Sure and don't say here are numbers. Just start with a number."
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    import random

    random.seed(42)
    total_requests = 100

    # Generate random batch sizes that sum to 100
    batch_sizes = []
    remaining = total_requests
    while remaining > 0:
        # Random batch size between 1 and min(10, remaining)
        batch_size = random.randint(1, min(10, remaining))
        batch_sizes.append(batch_size)
        remaining -= batch_size

    print(f"Random batch sizes: {batch_sizes}")
    print(f"Total batches: {len(batch_sizes)}, Total requests: {sum(batch_sizes)}")

    start_time = time.time()
    all_outputs = []

    for batch_idx, batch_size in enumerate(batch_sizes):
        prompts = [formatted_prompt] * batch_size
        results = llm.generate(prompts, sampling_params, use_tqdm=False)
        all_outputs.extend([result['text'] for result in results])

        if (batch_idx + 1) % max(1, len(batch_sizes) // 10) == 0:
            print(f"  Progress: {len(all_outputs)}/{total_requests} requests ({batch_idx + 1}/{len(batch_sizes)} batches)")

    end_time = time.time()

    # Save outputs
    for i, output_text in enumerate(all_outputs):
        with open(os.path.join(output_dir, f"output_{i+1}.txt"), "w") as f:
            f.write(output_text)

    # Analyze results
    unique_outputs = set(all_outputs)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Time: {end_time - start_time:.2f}s")
    print(f"Total samples: {len(all_outputs)}")
    print(f"Unique samples: {len(unique_outputs)}")
    print(f"Batch sizes used: {batch_sizes}")
    print(f"Status: {'✓ PASS - All outputs identical' if len(unique_outputs) == 1 else f'✗ FAIL - Found {len(unique_outputs)} different outputs'}")

    # Print all different outputs
    if len(unique_outputs) > 1:
        print(f"\n{'='*60}")
        print("DIFFERENT OUTPUTS")
        print(f"{'='*60}")
        for i, output in enumerate(unique_outputs, 1):
            count = all_outputs.count(output)
            print(f"\nOutput #{i} (appeared {count} times):")
            print("-" * 60)
            print(output)
            print("-" * 60)

if __name__ == "__main__":
    main()
