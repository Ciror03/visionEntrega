import torch

def parse_labels(path_txt, img_w, img_h):
    gt_boxes = []
    gt_labels = []
    with open(path_txt, "r") as f:
        for linea in f:
            cls, xc, yc, ww, hh = map(float, linea.strip().split())
            xc *= img_w
            yc *= img_h
            ww *= img_w
            hh *= img_h
            x1 = xc - ww / 2
            y1 = yc - hh / 2
            x2 = xc + ww / 2
            y2 = yc + hh / 2
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(int(cls))
    return torch.tensor(gt_boxes), torch.tensor(gt_labels,dtype=torch.int64)