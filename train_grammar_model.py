from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
import torch

# ===========================
# 1️⃣ Load your dataset
# ===========================
train_df = pd.read_csv("test.csv")
val_df = pd.read_csv("validation.csv")

# Rename columns if necessary
train_df.columns = ['incorrect', 'corrected']
val_df.columns = ['incorrect', 'corrected']

# Save temporary parquet for HuggingFace Datasets
train_df.to_csv("train_temp.csv", index=False)
val_df.to_csv("val_temp.csv", index=False)

# ===========================
# 2️⃣ Load with HuggingFace Datasets
# ===========================
dataset = load_dataset('csv', data_files={
    'train': 'train_temp.csv',
    'validation': 'val_temp.csv'
})

# ===========================
# 3️⃣ Load model and tokenizer
# ===========================
model_name = "t5-small"   # lightweight model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ===========================
# 4️⃣ Tokenize function
# ===========================
def preprocess_function(examples):
    inputs = ["gec: " + (text if text is not None else "") for text in examples['incorrect']]
    targets = [text if text is not None else "" for text in examples['corrected']]
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# ===========================
# 5️⃣ Training arguments
# ===========================
training_args = TrainingArguments(
    output_dir="./grammar_model",
    eval_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False,
)

# ===========================
# 6️⃣ Define Trainer
# ===========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# ===========================
# 7️⃣ Train the model
# ===========================
trainer.train()

# ===========================
# 8️⃣ Save final model
# ===========================
trainer.save_model("./grammar_corrector_t5_small")
tokenizer.save_pretrained("./grammar_corrector_t5_small")

print("✅ Training complete! Model saved at ./grammar_corrector_t5_small")
