import warnings
import os
import datasets
import pandas as pd
from tqdm import tqdm
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

warnings.filterwarnings("ignore")

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-14B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized

    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

df_group = pd.read_csv("gruppe.csv")

# df = pd.read_excel("../dataset/company_accounting_item_10_groups.xlsx")
# Read all excel files in the dataset folder
df = pd.DataFrame()
for file in tqdm(os.listdir("data")):
    if file.endswith(".xlsx"):
        df_file = pd.read_excel(f"data/{file}")
        # Only keep company_no in df_group["gruppe"]
        df_file = df_file[df_file["company_no"].isin(df_group["gruppe"])]
        df = pd.concat([df, df_file])

df = df[["item_name", "account_name"]]
df.dropna(inplace=True)
print("Number of rows in the dataset:", len(df))

def create_conversation(row):
    item_name = row["item_name"]
    account_name = row["account_name"]

    user_input = (
        f'Please provide the "account_name" for the following "item_name" in JSON format:\n\n'
        f'"item_name": "{item_name}"'
    )

    assistant_input = (
        f'{{"item_name": "{item_name}", "account_name": "{account_name}"}}'
    )

    conversation = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_input}
    ]

    return conversation


df["text"] = df.apply(create_conversation, axis=1)

dataset = datasets.Dataset.from_pandas(df[["text"]])

system_msg = """
You are an AI language model trained to map various \"item_name\"s to their corresponding \"account_name\"s for accounting purposes. Your goal is to assist users by providing accurate account classifications based on the item names they provide.\n\n**Instructions:**\n\n- When given an \"item_name\", return the corresponding \"account_name\" in a JSON object.\n\n- The JSON object must include both the \"item_name\" provided by the user and the correct \"account_name\".\n\n- Always output the result strictly in JSON format without additional text or explanations.\n\n- If you are unsure of the correct \"account_name\" for a given \"item_name\", set the \"account_name\" value to \"Unknown\".\n\n- Use the knowledge you have been trained on to make the most accurate mappings possible.\n\n- Here is an example of the expected JSON format:\n\n{\"item_name\": \"Support Januar\", \"account_name\": \"Aufwendungen für Lizenzen, Konzessionen\"}
"""

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
    system_message=system_msg
)


def formatting_prompts_func(examples):
    conversation = examples["text"]
    # texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in conversation]
    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)[:-4] for c in conversation]
    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=1,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs = 4, # Set this for 1 full training run.
        # max_steps=5,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=50,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=19,
        output_dir="../test/outputs",
        report_to="none",
        save_total_limit=3,
        # load_best_model_at_end = True
    ),
)

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer_stats = trainer.train(
    # resume_from_checkpoint=True
)

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
