from ultralytics import YOLO

# Load a model

model = YOLO("yolov8s.pt") 
model.to('cuda')
# Use the model
model.train(data="SUAVData.yaml", epochs=10,batch=8)
metrics = model.val()
