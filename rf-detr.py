import os
import multiprocessing

# 환경 변수 설정
# Secrets removed for security

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    
    model.train(
        dataset_dir="C:/capstone/data/rf-detr_fixed_data",  # 새로 만든 데이터셋 경로
        epochs=15,
        batch_size=2,  # 작은 배치 크기로 시작
        grad_accum_steps=4,  # 누적 스텝 사용
        lr=1e-4,
        #num_workers=0,  # Windows에서는 0으로 설정
        seed=42
    )
# import supervision as sv
# from tqdm import tqdm
# from supervision.metrics import MeanAveragePrecision

# ds = sv.DetectionDataset.from_coco(
#     images_directory_path=f"{dataset.location}/test",
#     annotations_path=f"{dataset.location}/test/_annotations.coco.json",
# )

# path, image, annotations = ds[0]
# image = Image.open(path)

# detections = model.predict(image, threshold=0.5)

# text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
# thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

# bbox_annotator = sv.BoxAnnotator(thickness=thickness)
# label_annotator = sv.LabelAnnotator(
#     text_color=sv.Color.BLACK,
#     text_scale=text_scale,
#     text_thickness=thickness,
#     smart_position=True)

# annotations_labels = [
#     f"{ds.classes[class_id]}"
#     for class_id
#     in annotations.class_id
# ]

# detections_labels = [
#     f"{ds.classes[class_id]} {confidence:.2f}"
#     for class_id, confidence
#     in zip(detections.class_id, detections.confidence)
# ]

# annotation_image = image.copy()
# annotation_image = bbox_annotator.annotate(annotation_image, annotations)
# annotation_image = label_annotator.annotate(annotation_image, annotations, annotations_labels)

# detections_image = image.copy()
# detections_image = bbox_annotator.annotate(detections_image, detections)
# detections_image = label_annotator.annotate(detections_image, detections, detections_labels)

# sv.plot_images_grid(images=[annotation_image, detections_image], grid_size=(1, 2), titles=["Annotation", "Detection"])

# detections_images = []

# for i in range(9):
#     path, image, annotations = ds[i]
#     image = Image.open(path)

#     detections = model.predict(image, threshold=0.5)

#     text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
#     thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

#     bbox_annotator = sv.BoxAnnotator(thickness=thickness)
#     label_annotator = sv.LabelAnnotator(
#         text_color=sv.Color.BLACK,
#         text_scale=text_scale,
#         text_thickness=thickness,
#         smart_position=True)

#     detections_labels = [
#         f"{ds.classes[class_id]} {confidence:.2f}"
#         for class_id, confidence
#         in zip(detections.class_id, detections.confidence)
#     ]

#     detections_image = image.copy()
#     detections_image = bbox_annotator.annotate(detections_image, detections)
#     detections_image = label_annotator.annotate(detections_image, detections, detections_labels)

#     detections_images.append(detections_image)

# sv.plot_images_grid(images=detections_images, grid_size=(3, 3), size=(12, 12))

# targets = []
# predictions = []

# for path, image, annotations in tqdm(ds):
#     image = Image.open(path)
#     detections = model.predict(image, threshold=0.5)

#     targets.append(annotations)
#     predictions.append(detections)

# map_metric = MeanAveragePrecision()
# map_result = map_metric.update(predictions, targets).compute()

# map_result.plot()

# confusion_matrix = sv.ConfusionMatrix.from_detections(
#     predictions=predictions,
#     targets=targets,
#     classes=ds.classes
# )

# confusion_matrix.plot()