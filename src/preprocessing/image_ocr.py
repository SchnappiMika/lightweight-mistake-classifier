# -*- coding: utf-8 -*-
"""
Image OCR module - PaddleOCR wrapper
"""

class ImageOCR:
    def __init__(self):
        self.available = False
        self._try_init()

    def _try_init(self):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            self.available = True
        except Exception as e:
            self.available = False

    def extract_text(self, image_path):
        if not self.available:
            return []
        try:
            result = self.ocr.ocr(image_path, cls=True)
            texts = []
            for line in result:
                for item in line:
                    texts.append(item[1][0])
            return texts
        except Exception as e:
            return []