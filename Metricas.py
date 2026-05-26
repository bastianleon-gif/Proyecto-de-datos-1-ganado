from ultralytics import YOLO
model = YOLO('best.pt') # Considerar ruta hacia el modelo
metrics = model.val()  # Evalúa automáticamente en el conjunto de validación. Considerar ruta hacia data.yaml
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"Precisión: {metrics.box.p[0]:.4f}") # Agregado [0] para extraer el valor pues es un arreglo
print(f"Recall: {metrics.box.r[0]:.4f}") # Agregado [0] para extraer el valor pues es un arreglo
print("="*30)