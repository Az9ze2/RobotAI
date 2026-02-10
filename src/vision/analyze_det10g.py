"""
Analyze det_10g.onnx raw outputs to reverse-engineer the format
"""
import cv2
import sys
import numpy as np
from pathlib import Path
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent / "src"))

def analyze_model_outputs():
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open webcam")
        return
    
    # Read a frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame")
        return
    
    print(f"Original frame shape: {frame.shape}")
    h, w = frame.shape[:2]
    
    # Preprocess like the detector does
    input_size = (640, 640)
    scale = min(input_size[0] / w, input_size[1] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    print(f"Scale factor: {scale}")
    print(f"Resized to: {new_w}x{new_h}")
    
    resized = cv2.resize(frame, (new_w, new_h))
    padded = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # Convert to model input
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    transposed = normalized.transpose(2, 0, 1)
    batched = np.expand_dims(transposed, axis=0)
    
    # Run model
    print("\nRunning inference...")
    session = ort.InferenceSession("models/buffalo_l/det_10g.onnx", 
                                   providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    
    outputs = session.run(output_names, {input_name: batched})
    
    print(f"\nModel has {len(outputs)} outputs")
    
    # Analyze each output
    for i in range(9):
        print(f"\n--- Output {i} ---")
        print(f"Shape: {outputs[i].shape}")
        print(f"Min: {outputs[i].min():.4f}")
        print(f"Max: {outputs[i].max():.4f}")
        print(f"Mean: {outputs[i].mean():.4f}")
        print(f"Std: {outputs[i].std():.4f}")
        
        # Show some sample values
        flat = outputs[i].flatten()
        print(f"Sample values (first 10): {flat[:10]}")
        
        # Find high-confidence detections (for score outputs)
        if i < 3:  # Score outputs
            high_conf = np.where(flat > 0.5)[0]
            print(f"High confidence (>0.5) count: {len(high_conf)}")
            if len(high_conf) > 0:
                print(f"High conf indices: {high_conf[:5]}")
                print(f"High conf scores: {flat[high_conf[:5]]}")
                
                # Check corresponding bbox and kps
                bbox_idx = i + 3
                kps_idx = i + 6
                
                print(f"\nCorresponding BBoxes (output {bbox_idx}):")
                for idx in high_conf[:3]:
                    bbox = outputs[bbox_idx][idx]
                    print(f"  Index {idx}: {bbox}")
                
                print(f"\nCorresponding KPS (output {kps_idx}):")
                for idx in high_conf[:3]:
                    kps = outputs[kps_idx][idx]
                    print(f"  Index {idx}: {kps}")

if __name__ == "__main__":
    analyze_model_outputs()
