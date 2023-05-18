import cv2
import numpy as np
#import pafy

# Configurações da rede YOLOv4
model_config = 'yolov4.cfg'
model_weights = 'yolov4.weights'
classes_file = 'coco.names'

# Carrega as classes do arquivo de nomes
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Configura a rede YOLOv4
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Carrega o vídeo
#video_file = 'video.mp4'+++
cap = cv2.VideoCapture(0)

new_width = 640
new_height = 480

# Loop pelos frames do vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (new_width, new_height))
    # Detecta objetos no frame usando a rede YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Processa as detecções
    conf_threshold = 0.70

    # Definir um valor de limiar de supressão para filtrar bounding boxes sobrepostas
    nms_threshold = 0.5

    # Processar as detecções
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar o non-maximum suppression para obter apenas as bounding boxes não redundantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Verificar se existem detecções após o NMS
    if len(indices) > 0:
        # Iterar sobre as bounding boxes selecionadas após o NMS
        for i in indices.flatten():
            left, top, width, height = boxes[i]
            class_id = class_ids[i]

            # Desenhar a bounding box e as informações relevantes no frame
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
            label = f'{classes[class_id]}: {confidences[i]:.2f}'
            cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar o resultado do processamento
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()