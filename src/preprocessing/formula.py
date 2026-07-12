# -*- coding: utf-8 -*-
"""
Formula recognition module - pix2tex (LaTeX-OCR) wrapper
"""
import os, subprocess, tempfile

class FormulaRecognizer:
    def __init__(self):
        self.available = False
        self._try_init()

    def _try_init(self):
        try:
            from pix2tex.cli import LatexOCR
            self.model = LatexOCR()
            self.available = True
        except Exception as e:
            self.available = False

    def recognize(self, image_path):
        if not self.available:
            return None
        try:
            from PIL import Image
            img = Image.open(image_path)
            latex = self.model(img)
            return latex
        except Exception as e:
            return None

    def formula_to_keywords(self, latex):
        """Convert LaTeX formula to keywords for classification"""
        if not latex:
            return []
        kw = []
        import re
        for token in re.findall(r'\\\\?[a-zA-Z]+|\\d+|[a-zA-Z]', latex):
            kw.append(token)
        return kw