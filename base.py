from litgpt import LLM
import time

# Your code block here

llm = LLM.load("meta-llama/Meta-Llama-3-8B-Instruct")
# text= llm.generate(prompt="Write me an essay on volcanos", max_new_tokens=4000)
# print(text)
# print(bench_d)



# 1) load your model

# 2) run benchmark (e.g. average over 3 runs)
start_time = time.perf_counter()
text = llm.generate(
    # num_iterations=1,
    prompt='''
    You are a professional storyteller. Your task: Write an imaginative short story that is approximately **4000 words** in length.

**Structure & Constraints:**
- **Characters**: Introduce three main characters:
  1. A curious child named Arin.
  2. A wise but mysterious forest spirit called Sylva.
  3. A playful animal companion, a fox named Rune.
- **Plot Outline**:
  - **Exposition (approx. 200 tokens)**: Set the scene—describe the setting and introduce Arin, Sylva, and Rune.
  - **Rising Action (approx. 300 tokens)**: Present a problem or conflict that arises.
  - **Climax (approx. 200 tokens)**: Describe the turning point of the story.
  - **Falling Action & Resolution (approx. 300 tokens)**: Wrap up the conflict and conclude the story.
- **Style & Tone**:
  - Use vivid, lyrical language that evokes wonder and nature.
  - Incorporate one short dialog (2-3 lines) between Arin and Sylva.
  - Keep paragraphs concise (3-5 sentences each).

**Additional Instructions:**
- After completing the story, provide an estimate: “Estimated token count: ___ tokens.”

Begin the story now.

    ''',
    top_k=1,
    max_new_tokens = 4000,
)

print(text)

end_time = time.perf_counter()
elapsed = end_time - start_time

print(f"Elapsed time: {elapsed:.6f} seconds")
print(f"Throughput: {elapsed/4000} tok/s")

# 3) inspect throughput
# print(f"Output: {text}\n")
# print("Throughput details:")
# for metric, values in stats.items():
#     print(f"  {metric}: {values!r}")

# # e.g. tokens/sec averaged over the runs:
# avg_tps = sum(stats["Inference speed in tokens/sec"]) / len(stats["Inference speed in tokens/sec"])
# print(f"\n→ {avg_tps:.1f} tokens/sec")
