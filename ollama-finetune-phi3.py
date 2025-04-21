from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

model_path = "/home/TomAdmin/phi-3-mini-128k-instruct"  # <-- your local model path

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

def preprocess(example):
    prompt = example["instruction"]
    response = example["output"]
    text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
    return tokenizer(text, truncation=True, max_length=2048, padding="max_length")

dataset = load_dataset("json", data_files="hf_train.jsonl", split="train")
tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset.column_names
)

training_args = TrainingArguments(
    output_dir="./phi3-finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=10,
    fp16=False,  # Set True if you have a GPU with fp16 support
    bf16=torch.cuda.is_available(),  # Use bf16 if available
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./phi3-finetuned")
tokenizer.save_pretrained("./phi3-finetuned")