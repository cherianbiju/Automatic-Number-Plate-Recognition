def process_image(image):
    import ultralytics
    import cv2
    import pytesseract
    import re
    import pandas as pd

    model = ultralytics.YOLO("License-Plate-Data/training_results/numberplate/weights/best.pt")

    # OCR config (remove in cloud)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    results = model(image)

    plate_img = None

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate_img = image[y1:y2, x1:x2]

    if plate_img is None:
        return None, None, None, None

    # OCR
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    plate_text = re.sub(r'[^A-Z0-9]', '', text)

    # CSV
    df = pd.read_csv("data.csv")
    result = df[df['plate'] == plate_text]

    if not result.empty:
        name = result.iloc[0]['name']
        phone = result.iloc[0]['phone']
        return plate_img, plate_text, name, phone
    else:
        return plate_img, plate_text, None, None