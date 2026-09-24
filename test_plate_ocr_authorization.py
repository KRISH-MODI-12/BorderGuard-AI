from pathlib import Path
import cv2

from backend.ai.engine import engine

BASE = Path(__file__).resolve().parent
IMAGE = BASE / 'database' / 'demo_videos' / 'plate_test.jpg'

print('AUTHORIZED PLATE DATABASE')
for plate in sorted(engine.authorized_plates()):
    print(' ', plate)

if not IMAGE.exists():
    print('\nNo test image found at:')
    print(IMAGE)
    print('Place a plate image there or change IMAGE in this script.')
    raise SystemExit(0)

frame = cv2.imread(str(IMAGE))
if frame is None:
    raise RuntimeError(f'Cannot read image: {IMAGE}')

plates, texts, error, db = engine.plate_verify(frame, 0.25)

print('\nOCR / AUTHORIZATION RESULTS')
for p in plates:
    print({
        'ocr_text': p.get('ocr_text', ''),
        'ocr_confidence': p.get('ocr_confidence', 0.0),
        'normalized_plate': p.get('normalized_plate', ''),
        'authorized': p.get('authorized', False),
        'status': p.get('status', 'OCR UNREADABLE'),
        'detector_confidence': p.get('confidence', 0.0),
    })

print('\nOCR error:', error)
