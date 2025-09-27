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
    
    prompt = "introduce yourself"
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Sending 1000 requests...")
    start_time = time.time()

    first_output_text = None
    all_outputs_are_identical = True

    for i in range(1000):
        # llm.generate expects a list of prompts
        outputs = llm.generate([formatted_prompt], sampling_params)
        current_output_text = outputs[0]['text']

        # Save output to file
        with open(os.path.join(output_dir, f"output_{i+1}.txt"), "w") as f:
            f.write(current_output_text)

        # Check for consistency
        if i == 0:
            first_output_text = current_output_text
        elif current_output_text != first_output_text:
            all_outputs_are_identical = False
        
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/1000 requests processed")


    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f} seconds.")

    if all_outputs_are_identical:
        print("\nSuccess: All 1000 generated outputs were identical.")
    else:
        print("\nFailure: Not all generated outputs were identical.")
        print("You can manually inspect the files in the 'output/' directory.")

if __name__ == "__main__":
    main()
