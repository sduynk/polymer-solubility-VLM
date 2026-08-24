import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import gc
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import supervision as sv
import torch
import torch._dynamo

from transformers import PaliGemmaProcessor, AutoModelForPreTraining
from peft import PeftModel, PeftConfig

torch._dynamo.config.suppress_errors = True


class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self):
        entries = []
        with open(self.jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path).convert("RGB")
        return image, entry


def get_true_label_from_filename(image_name: str):
    lower_name = image_name.lower()
    if 'partialsoluble' in lower_name or 'partialsobule' in lower_name:
        return 'partialsoluble'
    elif 'insoluble' in lower_name:
        return 'insoluble'
    elif 'colloidal' in lower_name:
        return 'colloidal'
    elif 'soluble' in lower_name:
        return 'soluble'
    else:
        return 'unknown'


def get_predicted_label_from_detections(detections):
    # eski inference_list_paligemma2 mantığıyla aynı
    if len(detections) > 0:
        detected_classes = set(detections['class_name'])
    else:
        detected_classes = set()

    if 'laser' in detected_classes and 'solid' in detected_classes:
        predicted_label = 'partialsoluble'
    elif 'solid' in detected_classes:
        predicted_label = 'insoluble'
    elif 'laser' in detected_classes:
        predicted_label = 'colloidal'
    else:
        predicted_label = 'soluble'

    return predicted_label, detected_classes


def evaluate_checkpoint(
    checkpoint_name,
    model_id,
    checkpoint_path,
    test_jsonl,
    image_dir,
    torch_dtype=torch.float16,
    max_new_tokens=256,
):
    print("\n" + "=" * 90)
    print(f"Evaluating: {checkpoint_name}")
    print(f"Model ID: {model_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 90)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    train_dataset = JSONLDataset(
        jsonl_file_path=test_jsonl.replace("_annotations.test.jsonl", "_annotations.train.jsonl"),
        image_directory_path=image_dir,
    )

    test_dataset = JSONLDataset(
        jsonl_file_path=test_jsonl,
        image_directory_path=image_dir,
    )

    CLASSES = train_dataset[0][1]['prefix'].replace("detect ", "").split(" ; ")

    config = PeftConfig.from_pretrained(checkpoint_path)
    base_model = AutoModelForPreTraining.from_pretrained(model_id)
    model = PeftModel.from_pretrained(base_model, checkpoint_path).to(DEVICE)
    processor = PaliGemmaProcessor.from_pretrained(model_id)

    model.eval()

    correct_count = 0
    total_count = 0
    records = []

    class_names = ["soluble", "colloidal", "partialsoluble", "insoluble"]
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}

    for i in tqdm(range(len(test_dataset))):
        image, label = test_dataset[i]
        image_name = label["image"]
        prefix = "<image>" + label["prefix"]

        inputs = processor(
            text=prefix,
            images=image,
            return_tensors="pt"
        ).to(torch_dtype).to(DEVICE)

        prefix_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generation = generation[0][prefix_length:]
            decoded = processor.decode(generation, skip_special_tokens=True)

        # eski inference_list_paligemma2 mantığı
        w, h = image.size
        detections = sv.Detections.from_lmm(
            lmm='paligemma',
            result=decoded,
            resolution_wh=(w, h),
            classes=CLASSES
        )

        predicted_label, detected_classes = get_predicted_label_from_detections(detections)
        true_label = get_true_label_from_filename(image_name)

        is_correct = predicted_label == true_label
        total_count += 1
        if is_correct:
            correct_count += 1

        if true_label in class_total:
            class_total[true_label] += 1
            if is_correct:
                class_correct[true_label] += 1

        records.append({
            "checkpoint": checkpoint_name,
            "image": image_name,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "detected_classes": ";".join(sorted(detected_classes)),
            "raw_output": decoded,
            "correct": is_correct
        })

    accuracy = correct_count / total_count if total_count > 0 else 0

    print("\nRESULT")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total_count - correct_count}")
    print(f"Total: {total_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")

    class_results = {}
    print("\nClass-level accuracy")
    for cls in class_names:
        cls_total = class_total[cls]
        cls_correct = class_correct[cls]
        cls_acc = cls_correct / cls_total if cls_total > 0 else None
        class_results[cls] = cls_acc

        if cls_acc is None:
            print(f"{cls}: no samples")
        else:
            print(f"{cls}: {cls_correct}/{cls_total} = {cls_acc:.2%}")

    summary = {
        "checkpoint": checkpoint_name,
        "model_id": model_id,
        "checkpoint_path": checkpoint_path,
        "total": total_count,
        "correct": correct_count,
        "incorrect": total_count - correct_count,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "soluble_accuracy": class_results["soluble"],
        "colloidal_accuracy": class_results["colloidal"],
        "partialsoluble_accuracy": class_results["partialsoluble"],
        "insoluble_accuracy": class_results["insoluble"],
    }

    del model
    del base_model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, records


def main():
    base_dir = r"C:\gemma_solubility"

    # aynı test split ile 3 checkpoint karşılaştırılacak
    test_jsonl = os.path.join(base_dir, "dataset_resized_resplit", "_annotations.test.jsonl")
    image_dir = os.path.join(base_dir, "dataset_resized_resplit")

    checkpoints = [
        {
            "checkpoint_name": "40epoch_10b_224_resplit",
            "model_id": "google/paligemma2-10b-pt-224",
            "checkpoint_path": os.path.join(
                base_dir,
                "check_point",
                "paligemma2_OD_40epoch_10b_224_resplit",
                "checkpoint-6560",
            ),
            "torch_dtype": torch.float16,
        },
        {
            "checkpoint_name": "80epoch_3b_224",
            "model_id": "google/paligemma2-3b-pt-224",
            "checkpoint_path": os.path.join(
                base_dir,
                "check_point",
                "paligemma2_OD_80epoch_3b",
                "checkpoint-12800",
            ),
            "torch_dtype": torch.float16,
        },
        {
            "checkpoint_name": "80epoch_3b_448_resplit",
            "model_id": "google/paligemma2-3b-pt-448",
            "checkpoint_path": os.path.join(
                base_dir,
                "check_point",
                "paligemma2_OD_80epoch_3b_448_resplit",
                "checkpoint-6560",
            ),
            "torch_dtype": torch.float16,
        },
    ]

    all_summaries = []
    all_records = []

    for cfg in checkpoints:
        summary, records = evaluate_checkpoint(
            checkpoint_name=cfg["checkpoint_name"],
            model_id=cfg["model_id"],
            checkpoint_path=cfg["checkpoint_path"],
            test_jsonl=test_jsonl,
            image_dir=image_dir,
            torch_dtype=cfg["torch_dtype"],
        )
        all_summaries.append(summary)
        all_records.extend(records)

    summary_df = pd.DataFrame(all_summaries)
    records_df = pd.DataFrame(all_records)

    summary_path = os.path.join(base_dir, "vlm_checkpoint_accuracy_summary_oldlogic.csv")
    records_path = os.path.join(base_dir, "vlm_checkpoint_predictions_oldlogic.csv")

    summary_df.to_csv(summary_path, index=False)
    records_df.to_csv(records_path, index=False)

    print("\nSaved files:")
    print(summary_path)
    print(records_path)

    print("\nFinal summary:")
    print(summary_df)


if __name__ == "__main__":
    main()