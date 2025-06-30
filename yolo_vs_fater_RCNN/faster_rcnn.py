import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt



class FasterRCNN:
    def __init__(self, model, classes, device=None, lr=0.005):
        """
        Inicializa el modelo con tu set de clases.
        :param classes: lista de nombres de clases. El índice de cada una es su etiqueta.
        :param device: 'cuda' o 'cpu'
        :param lr: learning rate
        """
        self.classes = ["__background__"] + classes  # fondo siempre es clase 0
        self.num_classes = len(self.classes)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Cargar modelo base y reemplazar la cabeza
        self.model = model
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        self.model.to(self.device)
        


    def train(self,train_loader,num_epochs = 10):
        optimizer = optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

        for epoch in range(num_epochs):
            self.model.train()
            for images, targets in train_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1} - Loss: {losses.item():.4f}")

    def predict(self,test_loader):
        metric = MeanAveragePrecision(iou_type="bbox")
        self.model.eval()

        for images, targets in test_loader:
            images = [img.to(self.device) for img in images]
            with torch.no_grad():
                preds = self.model(images)

            # Adaptar predicciones
            preds_fmt = []
            for p in preds:
                preds_fmt.append({
                    "boxes": p["boxes"].cpu(),
                    "scores": p["scores"].cpu(),
                    "labels": p["labels"].cpu()
                })

            # Adaptar ground truths
            targets_fmt = []
            for t in targets:
                targets_fmt.append({
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu()
                })

            metric.update(preds_fmt, targets_fmt)

        results = metric.compute()
        return results
    
    def show_predictions(self, dataset, idx, label_map, score_thresh=0.5):
        self.model.eval()
        
        # Colores únicos por clase
        class_colors = {
            "cow": "red",
            "deer": "yellow",
            "horse": "blue"
        }

        # Obtener imagen original (sin transformar)
        image, _ = dataset[idx]
        image = image.to(self.device).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(image)[0]

        # Filtrar por score
        keep = prediction['scores'] > score_thresh
        boxes = prediction['boxes'][keep]
        labels = prediction['labels'][keep]
        scores = prediction['scores'][keep]

        # Preparar textos y colores
        label_texts = []
        colors = []
        for i, l in enumerate(labels):
            name = label_map[l.item()]
            label_texts.append(f"{name}: {scores[i]:.2f}")
            colors.append(class_colors[name])

        # Convertir imagen a uint8 para dibujar
        img_uint8 = (image[0] * 255).byte().cpu()
        if img_uint8.shape[0] == 1:
            img_uint8 = img_uint8.repeat(3, 1, 1)

        # Dibujar bounding boxes con textos
        img_boxes = draw_bounding_boxes(
            img_uint8,
            boxes.cpu(),
            labels=label_texts,
            colors=colors,
            width=2,
            font_size=60
        )

        # Mostrar imagen con leyenda de clases
        plt.figure(figsize=(10, 10))
        plt.imshow(F.to_pil_image(img_boxes))

        # Leyenda
        legend_handles = [
            plt.Line2D([0], [0], color=color, lw=4, label=cls)
            for cls, color in class_colors.items()
        ]
        plt.legend(handles=legend_handles, loc="upper right")
        plt.axis("off")
        plt.show()

    def get_prediction(self, dataset, idx, score_thresh=0.5):
        """
        Devuelve la predicción (sin visualización) para una imagen del dataset.
        """
        self.model.eval()
        image, _ = dataset[idx]
        image_input = image.to(self.device).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(image_input)[0]

        # Filtrar por score si querés
        keep = pred['scores'] > score_thresh
        boxes = pred['boxes'][keep]
        scores = pred['scores'][keep]
        labels = pred['labels'][keep]

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }, image
    