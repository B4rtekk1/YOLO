import os
import fiftyone as fo
import fiftyone.types as fot
import cv2
import torch
import numpy as np

# Import predictor from inference.py
from inference import YOLOv11Predictor, COCO_NAMES

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, "coco_mini")
    val_images_dir = os.path.join(dataset_dir, "val2017")
    val_anno_path = os.path.join(dataset_dir, "annotations", "instances_val2017.json")
    model_path = os.path.join(base_dir, "saves", "last.pt")
    
    name = "yolo_val_results"
    if name in fo.list_datasets():
        print(f"Loading existing dataset '{name}'...")
        dataset = fo.load_dataset(name)
    else:
        print(f"Creating new dataset '{name}' from {val_images_dir}...")
        dataset = fo.Dataset.from_dir(
            dataset_type=fot.COCODetectionDataset,
            data_path=val_images_dir,
            labels_path=val_anno_path,
            name=name,
        )
    
    # FiftyOne wczytuje COCO do pola 'detections'
    # Zmieńmy nazwę na 'ground_truth' dla czytelności
    if "detections" in dataset.get_field_schema():
        dataset.rename_sample_field("detections", "ground_truth")
    
    # Dodajmy tagi do próbek bez predykcji, żeby wiedzieć co przetworzyć
    if "predictions" not in dataset.get_field_schema():
        dataset.tag_samples("inference_pending")
    else:
        # Tagi tylko dla próbek bez predykcji
        view = dataset.exists("predictions", False)
        view.tag_samples("inference_pending")

    # OPCJONALNIE: Ogranicz do 100 obrazów, żeby nie czekać wieczność
    # view = dataset.head(100) 
    view = dataset # zostaw tak, jeśli chcesz wszystkie
    # view = dataset # zostaw tak, jeśli chcesz wszystkie

    print(f"Loading model: {model_path}")
    predictor = YOLOv11Predictor(
        weights=model_path,
        device="0" if torch.cuda.is_available() else "cpu",
        conf_thres=0.1, # Niższy próg, żeby na pewno coś zobaczyć
    )

    # Limit próbkowania dla wydajności (ustaw None, aby przetworzyć wszystko)
    MAX_SAMPLES = None  
    
    view = dataset.match_tags("inference_pending")
    if MAX_SAMPLES is not None:
        view = view.limit(MAX_SAMPLES)
    
    # Sprawdź czy są próbki do przetworzenia
    num_to_process = view.count()
    if num_to_process == 0:
        if "predictions" not in dataset.get_field_schema():
            # Jeśli brak pola predictions, weź pierwsze próbki
            view = dataset
            if MAX_SAMPLES is not None:
                view = view.limit(MAX_SAMPLES)
            num_to_process = view.count()
        else:
            print("Wszystkie próbki mają już predykcje. Pomijam inferencję.")

    if num_to_process > 0:
        print(f"Running inference on {num_to_process} samples...")
        with fo.ProgressBar() as pb:
            for sample in pb(view):
                # Używamy sample.filepath, upewniając się, że ścieżka jest poprawna dla Windows
                path = os.path.normpath(sample.filepath)
                
                if not os.path.exists(path):
                    print(f"FAILED: File does not exist: {path}")
                    continue

                # Robust image reading for Windows with non-ASCII paths
                image = None
                try:
                    with open(path, 'rb') as f:
                        image_data = f.read()
                    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                except Exception as e:
                    print(f"ERROR reading {path}: {e}")
                    continue

                if image is None:
                    print(f"FAILED to open image: {path}")
                    continue
                    
                h, w = image.shape[:2]
                results = predictor.predict(image)
                
                # Konwersja wyników na format FiftyOne
                detections = []
                for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
                    # YOLO: [x1, y1, x2, y2] -> FiftyOne: [x_rel, y_rel, w_rel, h_rel]
                    x1, y1, x2, y2 = box
                    rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
                    
                    detections.append(
                        fo.Detection(
                            label=COCO_NAMES[label] if label < len(COCO_NAMES) else str(label),
                            bounding_box=rel_box,
                            confidence=score
                        )
                    )
                
                sample["predictions"] = fo.Detections(detections=detections)
                sample.tags.remove("inference_pending") if "inference_pending" in sample.tags else None
                sample.save()

    print("Inference complete. Launching FiftyOne App...")
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()