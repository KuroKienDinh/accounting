import json

from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only


def formatting_prompts_func(examples):
    # examples["text"] is a list of JSON strings (one per line in the txt file)
    convos = []
    for record in examples["text"]:
        # Convert JSON string to a Python list of {role, content} dicts
        conversation = json.loads(record)
        convos.append(conversation)

    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}


if __name__ == "__main__":
    data_files = "../dataset/llm_data_mapping_item_promp_2.txt"
    output_dir = "../outputs/deepseek_r1_qwen"
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit"

    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    dataset = load_dataset("text", data_files=data_files)
    dataset = dataset["train"].map(formatting_prompts_func, batched=True, num_proc=1)

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=max_seq_length, load_in_4bit=load_in_4bit, dtype=None)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

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
            num_train_epochs=4,  # Set this for 1 full training run.
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
            output_dir=output_dir,
            report_to="none",
            save_total_limit=3,
            # load_best_model_at_end = True
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<｜User｜>",
        response_part="<｜Assistant｜>",
    )

    trainer_stats = trainer.train()
