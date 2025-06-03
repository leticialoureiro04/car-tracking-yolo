import time
import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

# Configurações
enable_gpu = False
model_file = "yolov8s.pt"
video_path = "video.mp4" 
video_output_path = "output_video.avi"
save_video = True
show_fps = True

conf = 0.3
iou = 0.3
max_det = 20
pixels_per_meter = 8  

tracker = "bytetrack.yaml"
track_args = {
    "persist": True,
    "verbose": False,
}

# Inicializar modelo
LOGGER.info("Inicializando modelo...")
model = YOLO(model_file, task="detect")
classes = model.names

# Abrir vídeo
cap = cv2.VideoCapture('C:\\Users\\Leticia\\Desktop\\EI 3.ºANO\\Aprendizagem Organizacional\\computer_vision\\ultralytics\\examples\\YOLO-Interactive-Tracking-UI\\BARULHO DE RODOVIA - BARULHO DE AVENIDA - SOM DE CARROS ACELERANDO - RELAXAR.mp4')
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(3)), int(cap.get(4))
vw = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if save_video else None

window_name = "YOLO Tracking"
cv2.namedWindow(window_name)

# Histórico de posições
previous_centers = {}

def get_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2

# Loop principal
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, conf=conf, iou=iou, max_det=max_det, tracker=tracker, **track_args)
    annotator = Annotator(frame)
    detections = results[0].boxes.data if results[0].boxes is not None else []

    for det in detections:
        det = det.tolist()
        if len(det) < 6:
            continue

        x1, y1, x2, y2 = map(int, det[:4])
        class_id = int(det[6]) if len(det) >= 7 else int(det[5])
        track_id = int(det[4]) if len(det) >= 6 else -1
        label = model.names[class_id]

        if label != "car":
            continue

        center = get_center(x1, y1, x2, y2)
        speed_kmh = 0.0

        if track_id in previous_centers:
            prev_center = previous_centers[track_id]
            dx, dy = center[0] - prev_center[0], center[1] - prev_center[1]
            pixel_distance = (dx**2 + dy**2) ** 0.5
            meters = pixel_distance / pixels_per_meter
            speed_kmh = meters * fps * 3.6  # m/s * 3.6 = km/h

        previous_centers[track_id] = center

        color = (0, 255, 0)
        annotator.box_label([x1, y1, x2, y2], f"car {speed_kmh:.1f} km/h", color=color)

    if show_fps:
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 25), 0, 0.7, (255, 255, 255), 1)

    cv2.imshow(window_name, frame)
    if save_video and vw:
        vw.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
if vw:
    vw.release()
cv2.destroyAllWindows()






