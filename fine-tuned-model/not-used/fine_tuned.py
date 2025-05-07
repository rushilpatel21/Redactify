import os
import json
import logging
import torch
import numpy as np
import evaluate # HF evaluate library
from datasets import (
    load_dataset,
    ClassLabel,
    Sequence,
    Features,
    Value,
    DatasetDict
)
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

# --- Configuration ---
# Base model from Hugging Face Hub
BASE_MODEL_NAME = "theekshana/deid-roberta-i2b2-NER-medical-reports"
# Path to your local dataset file (relative to script location)
DATASET_FILE = os.path.join("i2b2_dataset", "test.jsonl")
# Path where the fine-tuned model will be saved (relative to script location)
NEW_MODEL_SAVE_PATH = "./fine_tuned_model"

# Define the 15 target labels for the fine-tuned model
TARGET_LABEL_LIST = [
    'O',        # Outside
    'B-DATE',   # Beginning of Date
    'I-DATE',   # Inside of Date
    'B-HOSPITAL',# Beginning of Hospital
    'I-HOSPITAL',# Inside of Hospital
    'B-PATIENT', # Beginning of Patient Name
    'I-PATIENT', # Inside of Patient Name
    'B-DOCTOR',  # Beginning of Doctor Name
    'I-DOCTOR',  # Inside of Doctor Name
    'B-AGE',     # Beginning of Age
    'I-AGE',     # Inside of Age
    'B-ID',      # Beginning of ID
    'I-ID',      # Inside of ID
    'B-LOCATION',# Beginning of Location
    'I-LOCATION' # Inside of Location
]

# Training arguments
NUM_EPOCHS = 3
BATCH_SIZE = 8 # Adjust based on GPU memory
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
# Split ratio for the single file (using 10% for evaluation)
EVAL_SPLIT = 0.1
# Max sequence length for tokenizer
MAX_SEQ_LEN = 512

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MedicalNERFineTuning")

# --- Global variables for helper functions ---
label_feature = None
seqeval = None

# --- Helper Functions ---
# --- MODIFICATION: Refined clean_labels for batched=True and case-insensitivity ---
def clean_labels(batch):
    """Maps 'NA' labels (case-insensitive) to 'O' in a batch."""
    if "labels" not in batch or not isinstance(batch["labels"], list):
        logger.warning("Batch missing 'labels' list, skipping cleaning.")
        return batch

    cleaned_batch_labels = []
    # Ensure 'tokens' exists and is a list for length checks, provide default if missing
    batch_tokens = batch.get("tokens")
    if not isinstance(batch_tokens, list) or len(batch_tokens) != len(batch["labels"]):
        logger.warning("Batch 'tokens' list is missing or has incorrect length. Length checks might be inaccurate.")
        # Create a placeholder list of lists with potentially incorrect lengths if tokens are missing/malformed
        batch_tokens = [[] for _ in batch["labels"]]

    for i, labels_list in enumerate(batch["labels"]): # Iterate through each example's label list in the batch
        if not isinstance(labels_list, list):
             logger.warning(f"Item at index {i} in 'labels' batch is not a list, skipping cleaning for this item.")
             cleaned_batch_labels.append(labels_list) # Keep original if not a list
             continue

        # Get corresponding tokens for length check for this specific example
        tokens_list = batch_tokens[i]
        if not isinstance(tokens_list, list):
             logger.warning(f"Tokens for batch item {i} is not a list. Using empty list for length check.")
             tokens_list = [] # Default to empty list if tokens are weird for this item

        # Check token/label length consistency before cleaning
        if len(tokens_list) != len(labels_list):
             logger.warning(f"Token/Label length mismatch in batch example {i} before cleaning: {len(tokens_list)} vs {len(labels_list)}. Labels might be incorrect.")

        # Clean the labels for the current example, converting Nones or other types to string first
        # Using upper() for case-insensitivity for "NA"
        current_cleaned_labels = []
        for lab in labels_list:
            lab_str = str(lab) # Ensure it's a string
            if lab_str.upper() == "NA":
                current_cleaned_labels.append("O")
            else:
                # Ensure the label is actually in our target list, otherwise map to 'O'
                # This prevents errors if other unexpected string labels exist
                if lab_str not in TARGET_LABEL_LIST:
                    if lab_str != "O": # Avoid warning if it's already 'O'
                        logger.warning(f"Unexpected label '{lab_str}' found in batch example {i}. Mapping to 'O'.")
                    current_cleaned_labels.append("O")
                else:
                    current_cleaned_labels.append(lab_str)

        # Optional: Ensure final label list matches token list length if mismatch occurred
        if len(tokens_list) > 0 and len(tokens_list) != len(current_cleaned_labels):
            logger.warning(f"Adjusting cleaned label length ({len(current_cleaned_labels)}) for batch example {i} to match token length ({len(tokens_list)}).")
            target_len = len(tokens_list)
            if len(current_cleaned_labels) > target_len:
                current_cleaned_labels = current_cleaned_labels[:target_len]
            else:
                # Pad with 'O'
                current_cleaned_labels.extend(["O"] * (target_len - len(current_cleaned_labels)))

        cleaned_batch_labels.append(current_cleaned_labels)

    batch["labels"] = cleaned_batch_labels # Update the entire batch's labels column
    return batch
# --- END MODIFICATION ---


def tokenize_and_align_labels(examples, tokenizer):
    """Tokenizes text and aligns integer labels."""
    if label_feature is None:
        raise ValueError("label_feature must be set globally in main()")

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_SEQ_LEN
    )

    aligned_labels = []
    for i, label_ids_list in enumerate(examples["labels"]): # These are integer IDs now
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        current_aligned_label_ids = []

        original_word_count = max(idx for idx in word_ids if idx is not None) + 1 if any(idx is not None for idx in word_ids) else 0

        if len(label_ids_list) != original_word_count:
            logger.warning(f"Length mismatch during alignment: {len(label_ids_list)} integer labels vs {original_word_count} words derived from tokens. Alignment might be incorrect. Assigning -100.")
            aligned_labels.append([-100] * len(word_ids))
            continue

        for word_idx in word_ids:
            if word_idx is None:
                current_aligned_label_ids.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(label_ids_list):
                     current_aligned_label_ids.append(label_ids_list[word_idx])
                else:
                     logger.error(f"Word index {word_idx} out of bounds for label list (len {len(label_ids_list)}). Assigning -100.")
                     current_aligned_label_ids.append(-100)
            else:
                current_aligned_label_ids.append(-100)
            previous_word_idx = word_idx
        aligned_labels.append(current_aligned_label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

def compute_metrics(p):
    """Computes seqeval metrics (precision, recall, F1, accuracy)."""
    if label_feature is None or seqeval is None:
        raise ValueError("label_feature and seqeval must be set globally in main()")

    predictions_logits, labels = p
    predictions = np.argmax(predictions_logits, axis=2)

    # --- MODIFICATION: Cast prediction/label IDs to int before int2str ---
    # Convert IDs back to strings, ignoring -100 labels
    true_predictions = [
        [label_feature.int2str(int(p)) for (p, l) in zip(prediction, label) if l != -100] # Cast p to int
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_feature.int2str(int(l)) for (p, l) in zip(prediction, label) if l != -100] # Cast l to int
        for prediction, label in zip(predictions, labels)
    ]
    # --- END MODIFICATION ---

    # Filter out empty lists if any examples had only -100 labels after tokenization/truncation
    true_predictions = [p for p in true_predictions if p]
    true_labels = [l for l in true_labels if l]

    if not true_labels: # Handle case where all labels were -100 in the batch
         logger.warning("Batch contained no valid labels for metric calculation.")
         return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    try:
        results = seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=0)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    except Exception as e:
        logger.error(f"Error computing seqeval metrics: {e}")
        logger.error(f"True Predictions sample: {true_predictions[:2]}")
        logger.error(f"True Labels sample: {true_labels[:2]}")
        # Return default values in case of error during computation
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}


# --- Main Fine-Tuning Function ---
def main():
    global label_feature, seqeval
    logger.info("--- Starting Medical NER Model Fine-Tuning ---")

    logger.info(f"Base Model: {BASE_MODEL_NAME}")
    logger.info(f"Dataset File: {DATASET_FILE}")
    logger.info(f"Fine-tuned Model Save Path: {NEW_MODEL_SAVE_PATH}")
    logger.info(f"Target Labels ({len(TARGET_LABEL_LIST)}): {TARGET_LABEL_LIST}")
    logger.info(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    logger.info(f"Eval Split: {EVAL_SPLIT}, Max Seq Len: {MAX_SEQ_LEN}")

    os.makedirs(NEW_MODEL_SAVE_PATH, exist_ok=True)

    logger.info(f"Loading dataset from {DATASET_FILE}...")
    if not os.path.exists(DATASET_FILE):
        logger.error(f"Dataset file not found at {DATASET_FILE}. Please ensure it exists.")
        return
    try:
        raw_dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
        logger.info(f"Dataset loaded successfully. Features inferred: {raw_dataset.features}")
        if 'tokens' not in raw_dataset.column_names or 'labels' not in raw_dataset.column_names:
             logger.error("Dataset loaded, but required 'tokens' or 'labels' column is missing.")
             return
    except Exception as e:
        logger.error(f"Failed to load dataset file {DATASET_FILE}: {e}", exc_info=True)
        return

    logger.info("Cleaning labels ('NA' -> 'O')...")
    try:
        # Apply the refined clean_labels function
        raw_dataset = raw_dataset.map(clean_labels, batched=True)
        logger.info("Label cleaning complete.")
        # Log a sample after cleaning to verify 'NA' removal
        if len(raw_dataset) > 0:
             logger.info(f"Example labels after cleaning (first 20): {raw_dataset[0]['labels'][:20]}")
    except Exception as e:
        logger.error(f"Failed during label cleaning: {e}", exc_info=True)
        return

    logger.info("Defining features with ClassLabel and casting dataset...")
    try:
        label_feature = ClassLabel(names=TARGET_LABEL_LIST) # Set global

        current_features = raw_dataset.features
        features_dict = {
            "tokens": Sequence(Value("string")),
            "labels": Sequence(label_feature), # Use ClassLabel here
        }
        for col_name, col_feature in current_features.items():
            if col_name not in features_dict:
                features_dict[col_name] = col_feature

        final_features = Features(features_dict)

        # Cast the dataset. This converts string labels to integer IDs
        raw_dataset = raw_dataset.cast(final_features) # Error occurred here previously
        logger.info(f"Dataset cast complete. Features: {raw_dataset.features}")
        if not isinstance(raw_dataset.features['labels'].feature, ClassLabel):
             logger.error("Casting failed: 'labels' column is not ClassLabel type.")
             return
        logger.info(f"Example labels after cast (should be integers): {raw_dataset[0]['labels'][:10]}")

    except Exception as e:
        # Log the specific label causing the issue if possible (though cast error might not provide it directly)
        logger.error(f"Failed during feature definition or casting: {e}", exc_info=True)
        # Add a check to see if 'NA' still exists somehow (shouldn't if cleaning worked)
        try:
            all_labels = set(l for ex in raw_dataset['labels'] for l in ex)
            logger.error(f"Unique labels found *after* cleaning attempt: {all_labels}")
        except:
            logger.error("Could not retrieve labels after casting error.")
        return

    logger.info(f"Splitting dataset ({1-EVAL_SPLIT:.0%} train / {EVAL_SPLIT:.0%} test)...")
    try:
        split_datasets = raw_dataset.train_test_split(test_size=EVAL_SPLIT, seed=42, shuffle=True)
        logger.info(f"Dataset splits created: {split_datasets}")
        if len(split_datasets["train"]) == 0 or len(split_datasets["test"]) == 0:
            logger.warning("One or both dataset splits are empty after splitting.")
    except Exception as e:
        logger.error(f"Failed to split dataset: {e}", exc_info=True)
        return

    logger.info(f"Loading tokenizer for {BASE_MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
        if not tokenizer.is_fast:
            logger.warning("Loaded a slow tokenizer.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
        return

    logger.info("Loading base model configuration and modifying for target labels...")
    try:
        target_label2id = {label: i for i, label in enumerate(TARGET_LABEL_LIST)}
        target_id2label = {i: label for i, label in enumerate(TARGET_LABEL_LIST)}
        num_target_labels = len(TARGET_LABEL_LIST)

        config = AutoConfig.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=num_target_labels,
            id2label=target_id2label,
            label2id=target_label2id
        )
        model = AutoModelForTokenClassification.from_pretrained(
            BASE_MODEL_NAME,
            config=config,
            ignore_mismatched_sizes=True
        )
        logger.info("Base model loaded with correctly sized classification head.")
    except Exception as e:
        logger.error(f"Failed to load model or config: {e}", exc_info=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    logger.info("Applying tokenization and label alignment to dataset splits...")
    try:
        columns_to_remove = [col for col in split_datasets["train"].column_names if col not in ["tokens", "labels"]]
        logger.info(f"Columns to remove during tokenization: {columns_to_remove}")

        tokenized_datasets = split_datasets.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer),
            batched=True,
            remove_columns=columns_to_remove
        )
        logger.info("Tokenization and alignment complete.")
        if len(tokenized_datasets["train"]) > 0:
            logger.info(f"Tokenized training dataset example [0] keys: {tokenized_datasets['train'][0].keys()}")
            logger.info(f"Tokenized training dataset example [0] labels sample: {tokenized_datasets['train'][0]['labels'][:20]}")
        else:
            logger.warning("Training dataset is empty after tokenization.")
        if len(tokenized_datasets["test"]) == 0:
             logger.warning("Evaluation dataset is empty after tokenization.")

    except Exception as e:
        logger.error(f"Failed during tokenization/alignment: {e}", exc_info=True)
        return

    logger.info("Setting up training components (Data Collator, Training Arguments, Metrics)...")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    try:
        seqeval = evaluate.load("seqeval") # Set global
    except Exception as e:
        logger.error(f"Failed to load seqeval metric: {e}. Install with 'pip install evaluate seqeval'.", exc_info=True)
        return

    train_steps_per_epoch = len(tokenized_datasets["train"]) // BATCH_SIZE if len(tokenized_datasets["train"]) > 0 else 1
    logging_steps = max(1, train_steps_per_epoch // 10)

    training_args = TrainingArguments(
        output_dir=NEW_MODEL_SAVE_PATH,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{NEW_MODEL_SAVE_PATH}/logs",
        logging_strategy="steps",
        logging_steps=logging_steps,
        report_to="none",
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Cannot start training.")
        return
    if len(eval_dataset) == 0:
        logger.warning("Evaluation dataset is empty. Evaluation metrics might be zero.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Starting fine-tuning on {len(train_dataset)} examples...")
    try:
        train_result = trainer.train()
        logger.info("Fine-tuning finished.")
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return

    if len(eval_dataset) > 0:
        logger.info(f"Evaluating the best model on {len(eval_dataset)} evaluation examples...")
        try:
            eval_metrics = trainer.evaluate()
            logger.info(f"Evaluation Results: {eval_metrics}")
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
    else:
        logger.warning("Evaluation dataset was empty. Skipping final evaluation.")

    logger.info(f"Saving final tokenizer and config to {NEW_MODEL_SAVE_PATH} (best model checkpoint)...")
    try:
        tokenizer.save_pretrained(NEW_MODEL_SAVE_PATH)
        config.save_pretrained(NEW_MODEL_SAVE_PATH)

        label_map_path = os.path.join(NEW_MODEL_SAVE_PATH, "target_label_map.json")
        with open(label_map_path, 'w') as f:
            json.dump({"id2label": target_id2label, "label2id": target_label2id}, f, indent=2)

        logger.info("Tokenizer, config, and label map saved successfully.")
        logger.info(f"Best model checkpoint saved in a sub-directory within {NEW_MODEL_SAVE_PATH}")
        logger.info(f"You can load the fine-tuned model using path: {os.path.abspath(NEW_MODEL_SAVE_PATH)}")
    except Exception as e:
        logger.error(f"Error saving final tokenizer/config/label map: {e}", exc_info=True)

    logger.info("--- Fine-Tuning Script Completed ---")

if __name__ == "__main__":
    main()
