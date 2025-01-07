import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# Load dataset
with open("dataset.json") as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    "input": [ex["input"] for ex in data["examples"]],
    "output": [ex["output"] for ex in data["examples"]]
})

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Tokenize data
def preprocess_function(examples):
    return tokenizer(examples["input"], text_target=examples["output"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")