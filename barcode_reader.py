import os
import cv2
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import numpy as np
from pyzbar import pyzbar

# Supported barcode types (pyzbar)
BARCODE_TYPES = {
    'CODE39': 'Code 39',
    'CODE128': 'Code 128',
    'QRCODE': 'QR Code',
    'EAN13': 'EAN-13',
    'EAN8': 'EAN-8',
    'UPCA': 'UPC-A',
    'DATAMATRIX': 'Data Matrix',
    'PDF417': 'PDF417',
    'ITF': 'Interleaved 2 of 5',
}

def preprocess_image(image: np.ndarray) -> List[np.ndarray]:
    """Apply multiple preprocessing techniques"""
    versions = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. Original
    versions.append(gray)

    # 2. High contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    versions.append(clahe.apply(gray))

    # 3. Binary threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(thresh)

    # 4. Inverted
    versions.append(cv2.bitwise_not(gray))

    # 5. Morphological
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    versions.append(morphed)

    # 6. Sharpened
    blur = cv2.GaussianBlur(gray, (0,0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    versions.append(sharpened)

    return versions

def decode_barcodes(image: np.ndarray) -> List[Dict]:
    """Decode barcodes using pyzbar with preprocessing"""
    preprocessed = preprocess_image(image)
    all_barcodes = []

    for proc in preprocessed:
        barcodes = pyzbar.decode(proc, symbols=None)
        for barcode in barcodes:
            data = barcode.data.decode('utf-8', errors='ignore')
            barcode_type = barcode.type
            x, y, w, h = barcode.rect.left, barcode.rect.top, barcode.rect.width, barcode.rect.height

            # Convert polygon to bounding box
            points = barcode.polygon
            if len(points) > 4:
                hull = cv2.convexHull(np.array([p for p in points], dtype=np.float32))
                hull = list(map(tuple, np.squeeze(hull)))
            else:
                hull = points

            all_barcodes.append({
                'data': data,
                'type': barcode_type,
                'type_name': BARCODE_TYPES.get(barcode_type, barcode_type),
                'x': x, 'y': y, 'w': w, 'h': h,
                'polygon': hull
            })

    # Remove duplicates
    seen = set()
    unique = []
    for b in all_barcodes:
        key = (b['data'], b['type'], b['x'], b['y'])
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique

def draw_barcodes(image: np.ndarray, barcodes: List[Dict], output_path: str):
    """Draw bounding boxes and labels"""
    img = image.copy()
    for i, b in enumerate(barcodes):
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        color = (0, 255, 0)  # Green

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Draw polygon (more accurate for rotated codes)
        if len(b['polygon']) >= 3:
            pts = np.array(b['polygon'], np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        # Label
        label = f"{b['type_name']}: {b['data'][:30]}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    cv2.imwrite(output_path, img)

def process_image(
    img_path: str,
    output_dir: str,
    draw: bool
) -> List[Dict]:
    """Process one image"""
    try:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Failed to load image")

        barcodes = decode_barcodes(image)
        records = []

        print(f"\n{Path(img_path).name}: Found {len(barcodes)} barcode(s)")

        for i, b in enumerate(barcodes):
            record = {
                'filename': Path(img_path).name,
                'index': i + 1,
                'type': b['type_name'],
                'data': b['data'],
                'x': b['x'],
                'y': b['y'],
                'width': b['w'],
                'height': b['h']
            }
            records.append(record)
            print(f"  [{i+1}] {b['type_name']}: {b['data']}")

        if draw and barcodes:
            out_path = os.path.join(output_dir, f"scanned_{Path(img_path).name}")
            draw_barcodes(image, barcodes, out_path)
            print(f"  Saved annotated: {out_path}")

        if not barcodes:
            records.append({
                'filename': Path(img_path).name,
                'index': 0,
                'type': '', 'data': 'NO BARCODE',
                'x': '', 'y': '', 'width': '', 'height': ''
            })

        return records

    except Exception as e:
        print(f"Error: {img_path} -> {e}")
        return [{
            'filename': Path(img_path).name,
            'index': 0,
            'type': 'ERROR',
            'data': str(e),
            'x': '', 'y': '', 'width': '', 'height': ''
        }]

def main():
    parser = argparse.ArgumentParser(
        description="Windows Barcode Reader (1D & 2D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python read_barcodes.py image.jpg --draw
  python read_barcodes.py "C:\\Scans\\" -o results --csv all_barcodes.csv
        """
    )
    parser.add_argument("input", help="Image file or folder")
    parser.add_argument("-o", "--output", default="barcode_results", help="Output folder")
    parser.add_argument("--csv", default="barcodes.csv", help="CSV output file")
    parser.add_argument("--draw", action="store_true", help="Save images with boxes")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Find images
    if input_path.is_dir():
        images = list(input_path.glob("*.png")) + \
                 list(input_path.glob("*.jpg")) + \
                 list(input_path.glob("*.jpeg")) + \
                 list(input_path.glob("*.bmp")) + \
                 list(input_path.glob("*.tiff"))
    else:
        images = [input_path]

    if not images:
        print("No images found!")
        return

    all_records = []
    for img in images:
        records = process_image(str(img), str(output_dir), args.draw)
        all_records.extend(records)

    # Save CSV
    csv_path = Path(args.csv)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'index', 'type', 'data', 'x', 'y', 'width', 'height'
        ])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nDone! Results saved to:")
    print(f"  CSV: {csv_path.resolve()}")
    if args.draw:
        print(f"  Images: {output_dir.resolve()}")

if __name__ == "__main__":
    main()