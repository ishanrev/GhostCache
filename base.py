from litgpt import LLM
import time
import math
# Your code block here

llm = LLM.load("meta-llama/Meta-Llama-3-8B-Instruct")
# text= llm.generate(prompt="Write me an essay on volcanos", max_new_tokens=4000)
# print(text)
# print(bench_d)

with open("prompt.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 1) load your model
prompt = text + " Summarize each chapter from the chapters provided in this story"
prompt_length = math.floor(len(prompt) * 0.48)
print(f"This is the prompt length {prompt_length}")
prompt = prompt[:prompt_length]
# 2) run benchmark (e.g. average over 3 runs)
start_time = time.perf_counter()
text = llm.generate(
    # num_iterations=1,
    prompt=prompt,
    top_k=1,
    max_new_tokens = 2000,
)

print(text)

end_time = time.perf_counter()
elapsed = end_time - start_time

print(f"Elapsed time: {elapsed:.6f} seconds")
print(f"Throughput: {2000/elapsed} tok/s")

# 3) inspect throughput
# print(f"Output: {text}\n")
# print("Throughput details:")
# for metric, values in stats.items():
#     print(f"  {metric}: {values!r}")

# # e.g. tokens/sec averaged over the runs:
# avg_tps = sum(stats["Inference speed in tokens/sec"]) / len(stats["Inference speed in tokens/sec"])
# print(f"\n→ {avg_tps:.1f} tokens/sec")
