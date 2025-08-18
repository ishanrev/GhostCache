from litgpt import LLM

llm = LLM.load("meta-llama/Meta-Llama-3-8B-Instruct")
# text= llm.generate(prompt="Write me an essay on volcanos", max_new_tokens=1000)
# print(text)
# print(bench_d)



# 1) load your model

# 2) run benchmark (e.g. average over 3 runs)
text = llm.generate(
    # num_iterations=1,
    prompt='''Can you explain to me how to bake a cake''',
    top_k=1,
    max_new_tokens = 50,
)

# 3) inspect throughput
# print(f"Output: {text}\n")
# print("Throughput details:")
# for metric, values in stats.items():
#     print(f"  {metric}: {values!r}")

# # e.g. tokens/sec averaged over the runs:
# avg_tps = sum(stats["Inference speed in tokens/sec"]) / len(stats["Inference speed in tokens/sec"])
# print(f"\n→ {avg_tps:.1f} tokens/sec")
