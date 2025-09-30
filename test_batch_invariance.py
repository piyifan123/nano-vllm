import os
import time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

from batch_invariant_ops import set_batch_invariant_mode

def main():
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup model and tokenizer
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # Use temperature=0 for greedy and deterministic output, with logprobs enabled
    sampling_params = SamplingParams(temperature=0, max_tokens=1000, logprobs=True)

    prompt = "Generate 100 random numbers. Go directly into it, don't say Sure and don't say here are numbers. Just start with a number."
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    import random

    random.seed(42)
    total_requests = 100

    # Generate random batch sizes that sum to total_requests with all unique values
    batch_sizes = []
    remaining = total_requests
    used_sizes = set()

    while remaining > 0:
        # Random batch size between 1 and min(30, remaining)
        max_size = min(30, remaining)

        # Try to find a unique batch size
        attempts = 0
        while attempts < 100:
            batch_size = random.randint(1, max_size)
            if batch_size not in used_sizes:
                batch_sizes.append(batch_size)
                used_sizes.add(batch_size)
                remaining -= batch_size
                break
            attempts += 1
        else:
            # If we can't find a unique size after many attempts, just use remaining
            batch_sizes.append(remaining)
            remaining = 0

    print(f"Random batch sizes: {batch_sizes}")
    print(f"Total batches: {len(batch_sizes)}, Total requests: {sum(batch_sizes)}")

    results_comparison = {}

    # Test with both modes: invariant mode disabled and enabled
    for invariant_mode in [False, True]:
        mode_name = "WITH batch invariance" if invariant_mode else "WITHOUT batch invariance"
        print(f"\n{'='*60}")
        print(f"Testing {mode_name}")
        print(f"{'='*60}")

        start_time = time.time()
        all_outputs = []

        all_logprobs = []

        with set_batch_invariant_mode(invariant_mode):
            for batch_idx, batch_size in enumerate(batch_sizes):
                prompts = [formatted_prompt] * batch_size
                results = llm.generate(prompts, sampling_params, use_tqdm=False)
                all_outputs.extend([result['text'] for result in results])
                all_logprobs.extend([tuple(result['logprobs']) if result['logprobs'] else None for result in results])

                if (batch_idx + 1) % max(1, len(batch_sizes) // 10) == 0:
                    print(f"  Progress: {len(all_outputs)}/{total_requests} requests ({batch_idx + 1}/{len(batch_sizes)} batches)")

        end_time = time.time()

        # Save outputs
        mode_dir = os.path.join(output_dir, "with_invariance" if invariant_mode else "without_invariance")
        os.makedirs(mode_dir, exist_ok=True)
        for i, (output_text, logprobs) in enumerate(zip(all_outputs, all_logprobs)):
            with open(os.path.join(mode_dir, f"output_{i+1}.txt"), "w") as f:
                f.write(output_text)
            if logprobs:
                with open(os.path.join(mode_dir, f"logprobs_{i+1}.txt"), "w") as f:
                    f.write(str(logprobs))

        # Analyze results using logprobs (more precise than text comparison)
        unique_logprobs = set(all_logprobs)
        unique_outputs = set(all_outputs)

        print(f"\nResults for {mode_name}:")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Total samples: {len(all_outputs)}")
        print(f"  Unique text outputs: {len(unique_outputs)}")
        print(f"  Unique logprob sequences: {len(unique_logprobs)}")
        print(f"  Status (by logprobs): {'✓ PASS - All outputs identical' if len(unique_logprobs) == 1 else f'✗ FAIL - Found {len(unique_logprobs)} different outputs'}")

        # Store results for comparison
        results_comparison[mode_name] = {
            'time': end_time - start_time,
            'unique_text_count': len(unique_outputs),
            'unique_logprobs_count': len(unique_logprobs),
            'outputs': all_outputs,
            'logprobs': all_logprobs,
            'unique_outputs': unique_outputs
        }

        # Print all different outputs with their logprobs
        if len(unique_outputs) >= 1:
            print(f"\n  Output details for {mode_name}:")
            for i, output in enumerate(unique_outputs, 1):
                count = all_outputs.count(output)
                # Find the corresponding logprobs for this output
                output_idx = all_outputs.index(output)
                corresponding_logprobs = all_logprobs[output_idx]

                print(f"\n  Output #{i} (appeared {count} times):")
                print("  " + "-" * 58)
                print("  Text: " + output[:200] + ("..." if len(output) > 200 else ""))
                print("  Logprobs (first 10): " + str(corresponding_logprobs[:10]) + ("..." if len(corresponding_logprobs) > 10 else ""))
                print("  " + "-" * 58)

        # Print all unique logprob sequences if different from unique outputs
        if len(unique_logprobs) > len(unique_outputs):
            print(f"\n  Note: Found {len(unique_logprobs)} unique logprob sequences but only {len(unique_outputs)} unique text outputs")
            print(f"  This means some outputs have identical text but different probability distributions")

    # Final comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Batch sizes used: {batch_sizes}")
    print()
    print(f"{'Mode':<30} {'Unique Texts':<15} {'Unique Logprobs':<18} {'Time (s)':<12} {'Status'}")
    print("-" * 80)
    for mode_name, results in results_comparison.items():
        status = "PASS" if results['unique_logprobs_count'] == 1 else "FAIL"
        print(f"{mode_name:<30} {results['unique_text_count']:<15} {results['unique_logprobs_count']:<18} {results['time']:<12.2f} {status}")

if __name__ == "__main__":
    main()
