import os
import cv2
import numpy as np
import tempfile

from PIL import Image
from ultralytics import YOLO, SAM
from fastapi import FastAPI, UploadFile, File


# ============================================================
# MODEL CONFIGURATION
# ============================================================

YOLO_MODEL_PATH = "best.pt"
SAM_MODEL_PATH = "sam2.1_b.pt"

detector = YOLO(YOLO_MODEL_PATH)
sam = SAM(SAM_MODEL_PATH)

REFERENCE_SIZE_MM = 50


# ============================================================
# ARUCO REFERENCE MARKER
# ============================================================

def detect_reference_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_6X6_250
    )

    parameters = cv2.aruco.DetectorParameters()

    aruco_detector = cv2.aruco.ArucoDetector(
        dictionary,
        parameters
    )

    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if ids is None:
        return None

    for i, marker_id in enumerate(ids.flatten()):

        if marker_id == 23:
            points = corners[i][0]

            lengths = [
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[1] - points[2]),
                np.linalg.norm(points[2] - points[3]),
                np.linalg.norm(points[3] - points[0])
            ]

            return float(np.mean(lengths))

    return None


# ============================================================
# MASK MEASUREMENT
# ============================================================

def measure_mask(mask):

    mask_uint8 = (
        (mask > 0.5).astype(np.uint8) * 255
    )

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)

    return {
        "area_pixels": area,
        "width_pixels": w,
        "height_pixels": h,
        "diameter_pixels": max(w, h)
    }


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def point_inside_mask(x, y, mask):

    h, w = mask.shape

    x = int(round(x))
    y = int(round(y))

    if x < 0 or x >= w:
        return False

    if y < 0 or y >= h:
        return False

    return bool(mask[y, x])


# ============================================================
# SIZE CLASSIFICATION
# ============================================================

def local_market_size(diameter_mm):

    if diameter_mm is None:
        return "Not measured"

    if diameter_mm > 60:
        return "Extra Large"

    elif diameter_mm >= 40:
        return "Medium"

    elif diameter_mm >= 20:
        return "Small"

    else:
        return "Below 20 mm"


# ============================================================
# CONDITION / DEFECT HANDLING
# ============================================================

def determine_condition(record):

    defects = []

    if record["rotten"]:
        defects.append("Visible rot")

    if record["sprout"]:
        defects.append("Sprouting")

    if record["double_split"]:
        defects.append("Double/split")

    if len(defects) == 0:
        return "No detected visible defect", []

    return "Visible defect detected", defects


# ============================================================
# LOT STATISTICS
# ============================================================

def calculate_lot_statistics(records):

    total = len(records)

    if total == 0:
        return {}

    rotten = sum(
        r["rotten"]
        for r in records
    )

    sprout = sum(
        r["sprout"]
        for r in records
    )

    double_split = sum(
        r["double_split"]
        for r in records
    )

    defective = sum(
        bool(r["defects"])
        for r in records
    )

    return {
        "total_onions": total,
        "rotten": rotten,
        "sprouted": sprout,
        "double_split": double_split,
        "defective_onions": defective,
        "defective_count_percent": round(
            defective / total * 100,
            2
        ),
        "rotten_count_percent": round(
            rotten / total * 100,
            2
        )
    }


# ============================================================
# STANDARDS
# ============================================================

STANDARDS = {

    "local_market": {
        "name": "NHB Local Market Size Categories",
        "type": "size_classification"
    },

    "nhb_nasik": {
        "name": "NHB Export - Nasik / Saurashtra / Bellary / Poona",
        "minimum_diameter_mm": 20,
        "decay_max_percent": 2,
        "defective_weight_limit_percent": 10
    },

    "nhb_bangalore": {
        "name": "NHB Export - Bangalore",
        "minimum_diameter_mm": 15,
        "decay_max_percent": 2
    },

    "nhb_krishnapuram": {
        "name": "NHB Export - Krishnapuram",
        "minimum_diameter_mm": 15,
        "decay_max_percent": 2
    }
}


# ============================================================
# RULE-BASED ASSESSMENT
# ============================================================

def apply_rules(records, standard_id):

    standard = STANDARDS[standard_id]

    stats = calculate_lot_statistics(records)

    results = []

    for record in records:

        diameter = record["diameter_mm"]

        if diameter is None:
            size_status = "Size not measured"

        elif "minimum_diameter_mm" in standard:

            if diameter < standard["minimum_diameter_mm"]:
                size_status = "Below minimum"

            else:
                size_status = "Meets minimum size"

        else:
            size_status = record["size_category"]

        results.append({
            "onion_id": record["onion_id"],
            "diameter_mm": record["diameter_mm"],
            "size_status": size_status,
            "condition": record["condition"],
            "defects": record["defects"]
        })

    return results, stats


# ============================================================
# VERIFICATION
# ============================================================

def verification_status(records):

    if not records:
        return (
            "Verification failed: "
            "no valid onion measurements were produced."
        )

    return (
        "Image processing completed. "
        "Results are based on visible image evidence."
    )


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(records, standard_id):

    standard = STANDARDS[standard_id]

    graded, stats = apply_rules(
        records,
        standard_id
    )

    verification = verification_status(records)

    report = f"""
# Onion Quality Assessment

## Selected Standard

**{standard["name"]}**

## Lot Summary

| Parameter | Result |
|---|---:|
| Total onions detected | {stats["total_onions"]} |
| Visible rotten onions | {stats["rotten"]} |
| Sprouted onions | {stats["sprouted"]} |
| Double/split onions | {stats["double_split"]} |
| Onions with detected visible defects | {stats["defective_onions"]} |
| Defective count indicator | {stats["defective_count_percent"]}%

## Onion Measurements
"""

    for onion in graded:

        diameter_text = (
            f'{onion["diameter_mm"]} mm'
            if onion["diameter_mm"] is not None
            else "Not measured"
        )

        report += f"""

### Onion {onion["onion_id"]}

- Diameter: **{diameter_text}**
- Size result: **{onion["size_status"]}**
- Condition: **{onion["condition"]}**
- Defects: **{", ".join(onion["defects"]) if onion["defects"] else "None detected"}**
"""

    report += f"""

## Verification

**{verification}**

## Important

The AI performs visual detection and image-based size estimation.

Count-based defect percentages are visual indicators and must not be interpreted as weight-based regulatory percentages.

Hidden/internal defects cannot be reliably determined from an ordinary RGB image.
"""

    return report


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Onion Quality Grading API",
    description="AI-based onion quality assessment using YOLO11, SAM2 and OpenCV",
    version="1.0.0"
)


@app.get("/")
def home():

    return {
        "message": "Onion Quality Grading API is running",
        "status": "ok"
    }


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(
    image_path,
    standard_id="local_market"
):

    image = cv2.imread(image_path)

    if image is None:
        return None, None, "Could not read image."

    # --------------------------------------------------------
    # 1. YOLO DETECTION
    # --------------------------------------------------------

    detections = detector.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )[0]

    onion_boxes = []

    for box in detections.boxes:

        class_id = int(box.cls[0])

        class_name = detector.names[class_id]

        if class_name.lower().strip() == "onion":

            onion_boxes.append(
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

    if len(onion_boxes) == 0:

        return (
            None,
            None,
            "No onions detected by YOLO."
        )

    # --------------------------------------------------------
    # 2. ARUCO CALIBRATION
    #
    # Marker is OPTIONAL.
    # If marker 23 is present, diameter is estimated in mm.
    # If marker is absent, defect detection still continues,
    # but size/diameter is reported as Not measured.
    # --------------------------------------------------------

    marker_pixels = detect_reference_marker(image)

    if marker_pixels is not None:

        mm_per_pixel = (
            REFERENCE_SIZE_MM / marker_pixels
        )

    else:

        mm_per_pixel = None

    # --------------------------------------------------------
    # 3. SAM2 SEGMENTATION
    # --------------------------------------------------------

    sam_results = sam(
        image_path,
        bboxes=onion_boxes,
        verbose=False
    )

    sam_result = sam_results[0]

    if sam_result.masks is None:

        return (
            None,
            None,
            "Could not generate onion masks with SAM2."
        )

    masks = (
        sam_result.masks.data
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # 4. ONION MEASUREMENTS
    # --------------------------------------------------------

    records = []

    for i, mask in enumerate(masks):

        measurement = measure_mask(mask)

        if measurement is None:
            continue

        if mm_per_pixel is not None:

            diameter_mm = (
                measurement["diameter_pixels"]
                * mm_per_pixel
            )

            diameter_mm = round(
                diameter_mm,
                2
            )

            size_category = local_market_size(
                diameter_mm
            )

        else:

            diameter_mm = None
            size_category = "Not measured"

        records.append({
            "onion_id": len(records) + 1,
            "diameter_mm": diameter_mm,
            "size_category": size_category,
            "rotten": False,
            "sprout": False,
            "double_split": False
        })

    if len(records) == 0:

        return (
            None,
            None,
            "SAM2 returned masks, but no valid onion measurements could be calculated."
        )

    # --------------------------------------------------------
    # 5. DEFECT ASSOCIATION
    # --------------------------------------------------------

    valid_mask_index = 0

    for mask in masks:

        measurement = measure_mask(mask)

        if measurement is None:
            continue

        if valid_mask_index >= len(records):
            continue

        for box in detections.boxes:

            class_id = int(box.cls[0])

            class_name = detector.names[class_id]

            class_name_lower = (
                class_name
                .lower()
                .strip()
            )

            if class_name_lower == "onion":
                continue

            defect_box = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            cx, cy = box_center(
                defect_box
            )

            if point_inside_mask(
                cx,
                cy,
                mask
            ):

                if class_name_lower in [
                    "rotten",
                    "spoiled"
                ]:

                    records[
                        valid_mask_index
                    ]["rotten"] = True

                elif class_name_lower in [
                    "sprout",
                    "sprouted"
                ]:

                    records[
                        valid_mask_index
                    ]["sprout"] = True

                elif class_name_lower in [
                    "double_split",
                    "double split"
                ]:

                    records[
                        valid_mask_index
                    ]["double_split"] = True

        valid_mask_index += 1

    # --------------------------------------------------------
    # 6. CONDITION
    # --------------------------------------------------------

    for record in records:

        condition, defects = determine_condition(
            record
        )

        record["condition"] = condition
        record["defects"] = defects

    # --------------------------------------------------------
    # 7. LOT STATISTICS
    # --------------------------------------------------------

    lot_statistics = calculate_lot_statistics(
        records
    )

    # --------------------------------------------------------
    # 8. VERIFICATION
    # --------------------------------------------------------

    verification = verification_status(
        records
    )

    # --------------------------------------------------------
    # 9. MARKDOWN REPORT
    # --------------------------------------------------------

    report = generate_report(
        records,
        standard_id
    )

    # --------------------------------------------------------
    # 10. STRUCTURED API OUTPUT
    # --------------------------------------------------------

    calibration = {
        "reference_size_mm": REFERENCE_SIZE_MM,
        "marker_detected": marker_pixels is not None,
        "marker_pixels": (
            round(marker_pixels, 2)
            if marker_pixels is not None
            else None
        ),
        "mm_per_pixel": (
            round(mm_per_pixel, 4)
            if mm_per_pixel is not None
            else None
        )
    }

    result_data = {

        "success": True,

        "standard": standard_id,

        "calibration": calibration,

        "lot_summary": lot_statistics,

        "onions": records,

        "verification": verification,

        "limitations": [
            "Assessment is based on visible image evidence.",
            "Count-based defect percentages are visual indicators.",
            "They must not be interpreted as weight-based regulatory percentages.",
            "Hidden or internal defects cannot be reliably determined from ordinary RGB images."
        ]
    }

    # --------------------------------------------------------
    # 11. ANNOTATED IMAGE
    # --------------------------------------------------------

    annotated = detections.plot()

    annotated = Image.fromarray(
        annotated
    )

    return annotated, result_data, report


# ============================================================
# PREDICT API
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    suffix = os.path.splitext(
        file.filename
    )[1]

    if not suffix:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(
            await file.read()
        )

        image_path = temp.name

    try:

        try:

            annotated, result_data, report = process_image(
                image_path,
                standard_id="local_market"
            )

        except Exception as e:

            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}"
            }

        if result_data is None:

            return {
                "success": False,
                "error": report
            }

        return result_data

    finally:

        if os.path.exists(image_path):

            os.remove(image_path)
