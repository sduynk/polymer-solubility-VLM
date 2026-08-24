import os
import json
from PIL import Image
from tqdm import tqdm
import torch
import supervision as sv
from transformers import PaliGemmaProcessor, AutoModelForPreTraining
from peft import PeftModel, PeftConfig
# from inference_list_paligemma2 import JSONLDataset  # Use your actual Dataset class here


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
test_dataset = JSONLDataset(
    jsonl_file_path="./dataset_resized/_annotations.test.jsonl",
    image_directory_path="./dataset_resized",
)

# Load model and processor
MODEL_ID = "google/paligemma2-3b-pt-224"
checkpoint_path = "check_point/paligemma2_OD_augmented_dataset_40epoch/checkpoint-6400"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

config = PeftConfig.from_pretrained(checkpoint_path)
base_model = AutoModelForPreTraining.from_pretrained(MODEL_ID)
torch.cuda.empty_cache()
model = PeftModel.from_pretrained(base_model, checkpoint_path).to(DEVICE)
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)

# Get class names
CLASSES = test_dataset[0][1]["prefix"].replace("detect ", "").split(" ; ")

# Output file
output_path = "result.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)


def is_valid_prediction(decoded_str, valid_classes):
    """
    Checks that each object has exactly 4 <locXXXX> tokens + valid class
    """
    objects = [x.strip() for x in decoded_str.strip().split(';')]
    for obj in objects:
        tokens = obj.split()
        loc_tokens = [t for t in tokens if t.startswith("<loc")]
        class_name = tokens[-1] if tokens else ""
        if len(loc_tokens) != 4 or class_name not in valid_classes:
            return False
    return True

with open(output_path, "w") as f_out:
    for i in tqdm(range(len(test_dataset))):
        image, label = test_dataset[i]
        image_name = label["image"]
        prefix = "<image>" + label["prefix"]
        suffix = label["suffix"]
        w, h = image.size

        # Encode inputs
        inputs = processor(
            text=prefix,
            images=image,
            return_tensors="pt"
        ).to(DTYPE).to(DEVICE)
        prefix_len = inputs["input_ids"].shape[-1]

        # Generate output
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            gen = gen[0][prefix_len:]
            decoded = processor.decode(gen, skip_special_tokens=True)

        # ⚠️ Check if prediction is valid before parsing
        def is_valid_prediction(decoded_str, valid_classes):
            objects = [x.strip() for x in decoded_str.strip().split(';')]
            for obj in objects:
                tokens = obj.split()
                loc_tokens = [t for t in tokens if t.startswith("<loc")]
                class_name = tokens[-1] if tokens else ""
                if len(loc_tokens) != 4 or class_name not in valid_classes:
                    return False
            return True

        if not is_valid_prediction(decoded, CLASSES):
            print(f"[{i}] Skipped invalid prediction: {decoded}")
            continue

        # Parse detections
        gt_detections = sv.Detections.from_lmm(
            lmm="paligemma", result=suffix, resolution_wh=(w, h), classes=CLASSES
        )
        pred_detections = sv.Detections.from_lmm(
            lmm="paligemma", result=decoded, resolution_wh=(w, h), classes=CLASSES
        )

        # Format output
        def format_detections(dets):
            return [
                {"class": det_class, "bbox": [float(x) for x in box]}
                for det_class, box in zip(dets["class_name"], dets.xyxy)
            ]

        entry = {
            "image": image_name,
            "gt": format_detections(gt_detections),
            "pred": format_detections(pred_detections),
        }

        f_out.write(json.dumps(entry) + "\n")

        # Optional cleanup
        del image, inputs, gen, decoded, gt_detections, pred_detections
        torch.cuda.empty_cache()
