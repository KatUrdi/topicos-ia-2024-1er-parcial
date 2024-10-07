from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings
from shapely.geometry import Polygon, Point

SETTINGS = get_settings()


def match_gun_bbox(segment: list[int], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    # Crear un polígono para el segmento de la persona usando las coordenadas del bounding box
    x1, y1, x2, y2 = segment
    segment_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    closest_box = None
    min_distance = float('inf')

    # Iterar sobre cada bounding box de las armas y calcular la distancia mínima al segmento
    for bbox in bboxes:
        bx1, by1, bx2, by2 = bbox
        bbox_polygon = Polygon([(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)])
        distance = segment_polygon.distance(bbox_polygon)

        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            closest_box = bbox

    return closest_box

def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_img = image_array.copy()

    for label, bbox in zip(segmentation.labels, segmentation.boxes):
        x1, y1, x2, y2 = bbox
        if label == 'danger':
            color = (255, 0, 0)  # Rojo para 'danger'
        else:
            color = (0, 255, 0)  # Verde para 'safe'

        # Dibujar el bounding box del segmento
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

        # Agregar la etiqueta ("danger" o "safe") sobre la caja
        cv2.putText(
            annotated_img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    return annotated_img


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 100):
        results = self.seg_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        people_indexes = [
            i for i in range(len(labels)) if labels[i] == 0
        ]  

        segments = [
            [list(map(int, pt)) for pt in polygon]
            for i, polygon in enumerate(results.masks.xy)
            if i in people_indexes
        ]
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in people_indexes
        ]

        detection = self.detect_guns(image_array, threshold)
        gun_bboxes = detection.boxes
        labels_txt = []

        for segment, box in zip(segments, boxes):
           
            person_polygon = Polygon(segment)
            person_center = person_polygon.centroid
            person_point = Point(person_center.x, person_center.y)

            
            is_danger = False
            for gun_bbox in gun_bboxes:
                x1, y1, x2, y2 = gun_bbox
                gun_box_center = Point((x1 + x2) / 2, (y1 + y2) / 2)
                distance = person_point.distance(gun_box_center)
                if distance <= max_distance:
                    is_danger = True
                    break
                person_box = Polygon([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])
                gun_box = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

                if person_box.intersects(gun_box):
                    is_danger = True
                    break

            if is_danger:
                labels_txt.append("danger")
            else:
                labels_txt.append("safe")

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(segments),
            polygons=segments,
            boxes=boxes,
            labels=labels_txt,
        )