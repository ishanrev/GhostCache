from litgpt import LLM

llm = LLM.load("meta-llama/Meta-Llama-3-8B-Instruct")
text= llm.generate(prompt="Write me an   essay on volcanos", max_new_tokens=500)
print(text)
# print(bench_d)
