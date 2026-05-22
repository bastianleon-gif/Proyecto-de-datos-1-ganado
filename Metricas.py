from ultralytics import YOLO
model = YOLO('best.pt')
metrics = model.val()  # Evalúa automáticamente en el conjunto de validación
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"Precisión: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")   
