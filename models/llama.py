from transformers.models.llama.modeling_llama import *

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(MODEL_NAME)

print(model.config)

