from ultralytics import YOLO

# Load model
model = YOLO('yolov8n-face-lindevs.pt')

def detailed_model_analysis(model):
    print("=== DETAILED FACE DETECTOR ANALYSIS ===")
    
    # Akses yang benar ke model sequence
    model_seq = model.model.model if hasattr(model.model, 'model') else model.model
    detect_head = model_seq[-1]
    
    print(f"\n=== DETECT HEAD CONFIGURATION ===")
    print(f"Detect head type: {type(detect_head)}")
    print(f"Number of detection scales: {len(detect_head.cv2)}")
    
    for i, (cv2_scale, cv3_scale) in enumerate(zip(detect_head.cv2, detect_head.cv3)):
        print(f"\nScale {i}:")
        print(f"  Input channels to cv2: {cv2_scale[0].conv.in_channels}")
        print(f"  Bbox output channels: {cv2_scale[-1].out_channels}") 
        print(f"  Class output channels: {cv3_scale[-1].out_channels}")
        print(f"  Bbox convolution layers: {len(cv2_scale)}")
        print(f"  Class convolution layers: {len(cv3_scale)}")
    
    # Analisis output shapes
    print(f"\n=== OUTPUT SHAPE ANALYSIS ===")
    print(f"DFL configuration: {detect_head.dfl}")
    print(f"Number of classes: 1 (face only)")
    
    # Test dengan input sample
    print(f"\n=== INFERENCE TEST ===")
    import torch
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outputs = model(dummy_input)
        print(f"Output type: {type(outputs)}")
        if hasattr(outputs, 'boxes'):
            print(f"Number of detections: {len(outputs.boxes)}")
            if len(outputs.boxes) > 0:
                print(f"Boxes shape: {outputs.boxes.xyxy.shape}")
                print(f"Confidences: {outputs.boxes.conf}")

# Jalankan analisis
detailed_model_analysis(model)