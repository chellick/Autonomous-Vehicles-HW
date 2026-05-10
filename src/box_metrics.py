def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0


def match_image(gt_boxes, pred_boxes, iou_thresh=0.5, conf_thresh=0.25):
    pred_boxes = [b for b in pred_boxes if b.get("confidence", 1.0) >= conf_thresh]
    matched_gt = set()
    pairs = []
    for pb in pred_boxes:
        best_iou = 0
        best_gi = -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            v = iou_xyxy(pb["xyxy"], gb["xyxy"])
            if v > best_iou:
                best_iou = v
                best_gi = gi
        if best_iou >= iou_thresh and best_gi >= 0:
            matched_gt.add(best_gi)
            gb = gt_boxes[best_gi]
            kind = "tp" if pb["class_name"] == gb["class_name"] else "wrong_class"
            pairs.append({
                "kind": kind,
                "gt_class": gb["class_name"],
                "pred_class": pb["class_name"],
                "iou": best_iou,
                "confidence": pb.get("confidence"),
            })
        else:
            pairs.append({
                "kind": "fp",
                "gt_class": None,
                "pred_class": pb["class_name"],
                "iou": best_iou,
                "confidence": pb.get("confidence"),
            })
    for gi, gb in enumerate(gt_boxes):
        if gi not in matched_gt:
            pairs.append({
                "kind": "fn",
                "gt_class": gb["class_name"],
                "pred_class": None,
                "iou": None,
                "confidence": None,
            })
    return pairs


def aggregate(pairs_per_image):
    counts = {"tp": 0, "fp": 0, "fn": 0, "wrong_class": 0}
    for entry in pairs_per_image:
        pairs = entry["pairs"] if isinstance(entry, dict) and "pairs" in entry else entry
        for p in pairs:
            counts[p["kind"]] += 1
    return counts


def precision_recall_f1(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    wc = counts["wrong_class"]
    prec = tp / max(1, tp + fp + wc)
    rec = tp / max(1, tp + fn + wc)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}
