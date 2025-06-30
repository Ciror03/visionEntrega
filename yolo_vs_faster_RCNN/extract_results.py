from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from faster_rcnn import FasterRCNN
import torch
import pandas as pd
import os

def evaluar_precision_recall_frcnn(modelo, dataset, score_thresh=0.5, iou_thresh=0.5):
    """
    Calcula precisión y recall exactos para todo el dataset, usando el método get_prediction del modelo.
    
    Args:
        modelo: objeto que tiene el método modelo.get_prediction(dataset, idx, score_thresh)
        dataset: dataset de validación o test
        score_thresh: umbral de score para filtrar predicciones
        iou_thresh: umbral de IOU para considerar un match como TP

    Returns:
        precision, recall, TP, FP, FN
    """
    TP = 0
    FP = 0
    FN = 0

    for idx in range(len(dataset)):
        pred, _ = modelo.get_prediction(dataset, idx, score_thresh=score_thresh)
        gt = dataset[idx][1]  # target dict con "boxes" y "labels"
        pred_boxes = pred["boxes"].cpu()
        gt_boxes = gt["boxes"].cpu()

        if len(pred_boxes) == 0:
            FN += len(gt_boxes)
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        matched_pred = set()

        for i in range(ious.shape[0]):
            max_iou, j = ious[i].max(0)
            if max_iou >= iou_thresh and j.item() not in matched_gt:
                TP += 1
                matched_gt.add(j.item())
                matched_pred.add(i)
            else:
                FP += 1

        FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return precision, recall, TP, FP, FN

def load_model_and_store_data_in_csv(name,model_path,loader,dataset):
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=4)  # 3 clases + background
    classes = ['cow','deer','horse']
    loaded_model = FasterRCNN(model, classes)
    loaded_model.model.load_state_dict(torch.load(model_path))
    resultados = loaded_model.predict(loader)
    # Evaluar precisión y recall con la función que ya tenés
    precision, recall, _,_,_ = evaluar_precision_recall_frcnn(loaded_model, dataset)
    # # Imprimir resultados
    map_50 = round(resultados["map_50"].item(), 5)
    map_50_95 = round(resultados["map"].item(), 5)

    df_metrics = pd.DataFrame([{
        "Model": name,
        "Precision": f"{precision:.5f}",
        "Recall":  f"{recall:.5f}",
        "mAP50": map_50,
        "mAP50-95": map_50_95
    }])
    csv_path = "final_results.csv"
    primer_escritura = not os.path.exists(csv_path)
    df_metrics.to_csv(csv_path, mode='a', header=primer_escritura, index=False)

