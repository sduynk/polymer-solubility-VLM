import os
import json
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import supervision as sv
import torch
from transformers import PaliGemmaProcessor, AutoModelForPreTraining
from peft import PeftModel, PeftConfig

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self):
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
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
        image = Image.open(image_path)
        return image, entry

# Load dataset
train_dataset = JSONLDataset(
    jsonl_file_path="./dataset_resized/_annotations.train.jsonl",
    image_directory_path="./dataset_resized",
)
test_dataset = JSONLDataset(
    jsonl_file_path="./dataset_resized/_annotations.test.jsonl",
    image_directory_path="./dataset_resized",
)

CLASSES = train_dataset[0][1]['prefix'].replace("detect ", "").split(" ; ")

# Load model
# MODEL_ID = "google/paligemma2-3b-pt-448"

MODEL_ID = "google/paligemma2-10b-pt-224"
checkpoint_path = "check_point/paligemma2_OD_40epoch_10b_224_resplit/checkpoint-6560"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float16

config = PeftConfig.from_pretrained(checkpoint_path)
base_model = AutoModelForPreTraining.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, checkpoint_path).to(DEVICE)
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)

# Run inference
os.makedirs("inference_outputs_3", exist_ok=True)

correct_count = 0
total_count = 0


for i in range(len(test_dataset)):
    image, label = test_dataset[i]
    image_name = label["image"]
    prefix = "<image>" + label["prefix"]

    inputs = processor(
        text=prefix,
        images=image,
        return_tensors="pt"
    ).to(TORCH_DTYPE).to(DEVICE)

    prefix_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generation = generation[0][prefix_length:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        print(f"[{i}] {image_name}: {decoded}")

    # Parse detections
    w, h = image.size
    detections = sv.Detections.from_lmm(
        lmm='paligemma',
        result=decoded,
        resolution_wh=(w, h),
        classes=CLASSES
    )

    # Print detection results (skip confidence if not available)
    if detections.confidence is None:
        print(f"[{i}] {image_name}:")
        for det_class, box in zip(detections['class_name'], detections.xyxy):
            print(f"  → {det_class}: {box.tolist()}")
    else:
        print(f"[{i}] {image_name}:")
        for det_class, box, conf in zip(detections['class_name'], detections.xyxy, detections.confidence):
            print(f"  → {det_class}: {box.tolist()}, confidence: {conf:.3f}")

    # Annotate and save
    # Determine predicted label from detections
    detected_classes = set(detections['class_name']) if detections else set()
    if 'laser' in detected_classes and 'solid' in detected_classes:
        predicted_label = 'partialsoluble'
    elif 'solid' in detected_classes:
        predicted_label = 'insoluble'
    elif 'laser' in detected_classes:
        predicted_label = 'colloidal'
    else:
        predicted_label = 'soluble'

    # Extract ground truth from image filename
    lower_name = image_name.lower()
    if 'partialsoluble' in lower_name or 'partialsobule' in lower_name:
        true_label = 'partialsoluble'
    elif 'insoluble' in lower_name:
        true_label = 'insoluble'
    elif 'colloidal' in lower_name:
        true_label = 'colloidal'
    elif 'soluble' in lower_name:
        true_label = 'soluble'
    else:
        true_label = 'unknown'

    is_correct = predicted_label == true_label
    correctness_prefix = "True_" if is_correct else "False_"
    total_count += 1
    if is_correct:
        correct_count += 1

    # Annotate and save
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator(smart_position=True).annotate(annotated_image, detections)

    save_name = correctness_prefix + image_name
    save_path = os.path.join("inference_outputs_3", save_name)
    annotated_image.save(save_path)
    
accuracy = correct_count / total_count if total_count > 0 else 0
print(f"\n✅ Prediction Accuracy: {correct_count}/{total_count} = {accuracy:.2%}")

