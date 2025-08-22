import os
import cv2
import json
import argparse
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count
from functools import partial

def process_outputs(outputs, model, img_file, out_img_path, out_json_path, confidence):
    detections = []
    annotated_frame = outputs.plot()

    for box in outputs.boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        conf = float(box.conf[0].item())
        if conf < confidence:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": label,
            "confidence": round(conf, 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

    # here we are saving the  annotated image
    cv2.imwrite(out_img_path, annotated_frame)

    # Saveing  JSON log
    log = {"filename": img_file, "detections": detections}
    with open(out_json_path, "w") as jf:
        json.dump(log, jf, indent=2)


def run_batch_inference(model, img_paths, output_dir, json_dir, confidence):
    # batch inference
    outputs_list = model(img_paths, conf=confidence)  

    for img_path, outputs in zip(img_paths, outputs_list):
        img_file = os.path.basename(img_path)
        out_img_path = os.path.join(output_dir, img_file)
        out_json_path = os.path.join(json_dir, img_file.replace(".jpg", ".json"))
        process_outputs(outputs, model, img_file, out_img_path, out_json_path, confidence)


def main(args):
    os.makedirs(args.output, exist_ok=True)
    json_folder = os.path.join(args.output, "json_logs")
    os.makedirs(json_folder, exist_ok=True)

    model = YOLO(args.model)

    # Collect all images
    img_files = [f for f in os.listdir(args.input) if f.lower().endswith(".jpg")]
    img_paths = [os.path.join(args.input, f) for f in img_files]

    if args.multiprocessing:
        # Split workload across CPU cores
        num_workers = min(cpu_count(), args.workers)
        chunk_size = len(img_paths) // num_workers + 1
        chunks = [img_paths[i:i+chunk_size] for i in range(0, len(img_paths), chunk_size)]

        with Pool(processes=num_workers) as pool:
            pool.map(partial(run_batch_inference, model, output_dir=args.output,
                             json_dir=json_folder, confidence=args.confidence), chunks)
    else:
        # Simple batch inference
        batch_size = args.batch_size
        for i in range(0, len(img_paths), batch_size):
            batch = img_paths[i:i+batch_size]
            run_batch_inference(model, batch, args.output, json_folder, args.confidence)

    print(f"*****Images saved to '{args.output}', JSON logs in '{json_folder}'****")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Inference Script with JSON Logging (Batch + Multiprocessing)")
    parser.add_argument("--input", type=str, required=True, help="Path to input folder with .jpg images")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model .pt file")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold (default=0.25)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference (default=16)")
    parser.add_argument("--multiprocessing", action="store_true", help="Enable multiprocessing for inference")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for multiprocessing")
    
    args = parser.parse_args()
    main(args)
