from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, load_in_8bit=True, device_map="auto")
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"])
model = get_peft_model(model, lora_cfg)

ds = load_dataset("json", data_files={"train":"train.jsonl","validation":"eval.jsonl"})

def preprocess(batch):
    text = [f"### Instruction:\n{p}\n\n### Response:\n{r}" for p, r in zip(batch["prompt"], batch["response"])]
    toks = tokenizer(text, truncation=True, max_length=1024, padding="max_length")
    toks["labels"] = toks["input_ids"].copy()
    return toks

tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

args = TrainingArguments(
    output_dir="sft-lora",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    save_steps=1000
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"])
trainer.train()
model.save_pretrained("sft-lora")
