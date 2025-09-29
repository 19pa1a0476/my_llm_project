from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token="")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config
)
base_model = model

# Load with/without LoRA adapter
try:
    model = PeftModel.from_pretrained(base_model, "sft-lora")
    print("Loaded with LoRA adapter.")
except Exception:
    model = base_model
    print("Loaded base model only.")
