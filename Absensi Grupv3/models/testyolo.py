from ultralytics import YOLO

# Load model
model = YOLO('yolov8n-face-lindevs.pt')

def test_with_local_images(model, image_folder=".", conf=0.3):
    """Test dengan gambar yang ada di folder lokal"""
    print("=== LOCAL IMAGES TEST ===")
    
    import cv2
    import os
    import glob
    
    # Cari file gambar di folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = ['Gambar Test/test.jpg']
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, extension)))
        image_files.extend(glob.glob(os.path.join(image_folder, extension.upper())))
    
    print(f"Found {len(image_files)} image files")
    
    for image_path in image_files:
        try:
            print(f"\nTesting: {os.path.basename(image_path)}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"  ❌ Cannot load image")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"  Image shape: {image_rgb.shape}")
            
            # Run detection
            results = model(image_rgb, conf=conf, verbose=False)
            detections = results[0].boxes
            
            if len(detections) > 0:
                boxes = detections.xyxy.cpu().numpy()
                confidences = detections.conf.cpu().numpy()
                
                print(f"  ✅ Detected {len(boxes)} faces!")
                for j, (box, conf) in enumerate(zip(boxes, confidences)):
                    print(f"    Face {j+1}: confidence {conf:.4f}")
                
                # Save result dengan bounding boxes
                result_image = results[0].plot()
                output_path = f"result_{os.path.basename(image_path)}"
                cv2.imwrite(output_path, result_image)
                print(f"  Result saved as: {output_path}")
            else:
                print("  ❌ No faces detected")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

# Test dengan gambar di folder saat ini
test_with_local_images(model)