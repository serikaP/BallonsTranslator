import numpy as np
from typing import List
import os
import logging

try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    logging.warning(
        'PaddleOCR is not installed. Install it by following https://www.paddlepaddle.org.cn/en/install/quick?docurl'
    )

import cv2
import re

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

# Specify the path for storing PaddleOCR models
PADDLE_OCR_PATH = os.path.join('data', 'models', 'paddle-ocr')
# Set an environment variable to store PaddleOCR models
os.environ['PPOCR_HOME'] = PADDLE_OCR_PATH

@register_OCR('paddle_ocr')
class PaddleOCRModule(OCRBase):
    # Mapping language names to PaddleOCR codes
    lang_map = {
        'Chinese & English': 'ch',
        'English': 'en',
        'French': 'fr',
        'German': 'german',
        'Japanese': 'japan',
        'Korean': 'korean',
        'Chinese Traditional': 'chinese_cht',
        'Italian': 'it',
        'Spanish': 'es',
        'Portuguese': 'pt',
        'Russian': 'ru',
        'Ukrainian': 'uk',
        'Belarusian': 'be',
        'Telugu': 'te',
        'Saudi Arabia': 'sa',
        'Tamil': 'ta',
        'Afrikaans': 'af',
        'Azerbaijani': 'az',
        'Bosnian': 'bs',
        'Czech': 'cs',
        'Welsh': 'cy',
        'Danish': 'da',
        'Dutch': 'nl',
        'Norwegian': 'no',
        'Polish': 'pl',
        'Romanian': 'ro',
        'Slovak': 'sk',
        'Slovenian': 'sl',
        'Albanian': 'sq',
        'Swedish': 'sv',
        'Swahili': 'sw',
        'Tagalog': 'tl',
        'Turkish': 'tr',
        'Uzbek': 'uz',
        'Vietnamese': 'vi',
        'Mongolian': 'mn',
        'Arabic': 'ar',
        'Hindi': 'hi',
        'Uyghur': 'ug',
        'Persian': 'fa',
        'Urdu': 'ur',
        'Serbian (Latin)': 'rs_latin',
        'Occitan': 'oc',
        'Marathi': 'mr',
        'Nepali': 'ne',
        'Serbian (Cyrillic)': 'rs_cyrillic',
        'Bulgarian': 'bg',
        'Estonian': 'et',
        'Irish': 'ga',
        'Croatian': 'hr',
        'Hungarian': 'hu',
        'Indonesian': 'id',
        'Icelandic': 'is',
        'Kurdish': 'ku',
        'Lithuanian': 'lt',
        'Latvian': 'lv',
        'Maori': 'mi',
        'Malay': 'ms',
        'Maltese': 'mt',
        'Adyghe': 'ady',
        'Kabardian': 'kbd',
        'Avar': 'ava',
        'Dargwa': 'dar',
        'Ingush': 'inh',
        'Lak': 'lbe',
        'Lezghian': 'lez',
        'Tabassaran': 'tab',
        'Bihari': 'bh',
        'Maithili': 'mai',
        'Angika': 'ang',
        'Bhojpuri': 'bho',
        'Magahi': 'mah',
        'Nagpur': 'sck',
        'Newari': 'new',
        'Goan Konkani': 'gom',
    }

    params = {
        'language': {
            'type': 'selector',
            'options': list(lang_map.keys()),
            'value': 'English',  # Default language
            'description': 'Select the language for OCR',
        },
        'device': DEVICE_SELECTOR(),
        'use_angle_cls': {
            'type': 'checkbox',
            'value': False,
            'description': 'Enable angle classification for rotated text',
        },
        'ocr_version': {
            'type': 'selector',
            'options': ['PP-OCRv4', 'PP-OCRv3', 'PP-OCRv2', 'PP-OCR'],
            'value': 'PP-OCRv4',
            'description': 'Select the OCR model version',
        },
        'enable_mkldnn': {
            'type': 'checkbox',
            'value': False,
            'description': 'Enable MKL-DNN for CPU acceleration',
        },
        'det_limit_side_len': {
            'value': 960,
            'description': 'Maximum side length for text detection',
        },
        'rec_batch_num': {
            'value': 6,
            'description': 'Batch size for text recognition',
        },
        'drop_score': {
            'value': 0.5,
            'description': 'Confidence threshold for text recognition',
        },
        'text_case': {
            'type': 'selector',
            'options': ['Uppercase', 'Capitalize Sentences', 'Lowercase'],
            'value': 'Capitalize Sentences',
            'description': 'Text case transformation',
        },
        'output_format': {
            'type': 'selector',
            'options': ['Single Line', 'As Recognized'],
            'value': 'As Recognized',
            'description': 'Text output format',
        },
    }

    device = DEFAULT_DEVICE

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.language = self.params['language']['value']
        self.device = self.params['device']['value']
        self.use_angle_cls = self.params['use_angle_cls']['value']
        self.ocr_version = self.params['ocr_version']['value']
        self.enable_mkldnn = self.params['enable_mkldnn']['value']
        self.det_limit_side_len = self.params['det_limit_side_len']['value']
        self.rec_batch_num = self.params['rec_batch_num']['value']
        self.drop_score = self.params['drop_score']['value']
        self.text_case = self.params['text_case']['value']
        self.output_format = self.params['output_format']['value']
        self.model = None
        self._setup_logging()
        self._load_model()

    def _setup_logging(self):
        if self.debug_mode:
            logging.getLogger('ppocr').setLevel(logging.DEBUG)
            logging.getLogger('paddleocr').setLevel(logging.DEBUG)
            logging.getLogger('predict_system').setLevel(logging.DEBUG)
        else:
            logging.getLogger('ppocr').setLevel(logging.WARNING)
            logging.getLogger('paddleocr').setLevel(logging.WARNING)
            logging.getLogger('predict_system').setLevel(logging.WARNING)

    def _load_model(self):
        lang_code = self.lang_map[self.language]
        use_gpu = True if self.device == 'cuda' else False
        if self.debug_mode:
            self.logger.info(f"Load the PaddleOCR model for the language: {self.language} ({lang_code}), GPU: {use_gpu}")
        self.model = PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=lang_code,
            use_gpu=use_gpu,
            ocr_version=self.ocr_version,
            enable_mkldnn=self.enable_mkldnn,
            det_limit_side_len=self.det_limit_side_len,
            rec_batch_num=self.rec_batch_num,
            drop_score=self.drop_score,
            det_model_dir=os.path.join(PADDLE_OCR_PATH, lang_code, self.ocr_version, 'det'),
            rec_model_dir=os.path.join(PADDLE_OCR_PATH, lang_code, self.ocr_version, 'rec'),
            cls_model_dir=os.path.join(PADDLE_OCR_PATH, lang_code, self.ocr_version, 'cls') if self.use_angle_cls else None,
        )

    def ocr_img(self, img: np.ndarray) -> str:
        if self.debug_mode:
            self.logger.debug(f"Start OCR for image size: {img.shape}")
        result = self.model.ocr(img, det=True, rec=True, cls=self.use_angle_cls)
        if self.debug_mode:
            self.logger.debug(f"Recognition result: {result}")
        text = self._process_result(result)
        return text

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                cropped_img = img[y1:y2, x1:x2]
                try:
                    result = self.model.ocr(cropped_img, det=True, rec=True, cls=self.use_angle_cls)
                    text = self._process_result(result)
                    blk.text = text if text else ''
                    
                    if self.debug_mode:
                        self.logger.debug(f"Processing a block with coordinates: ({x1}, {y1}, {x2}, {y2})")
                        self.logger.debug(f"Text from the block ({x1}, {y1}, {x2}, {y2}): {text}")

                except Exception as e:
                    if self.debug_mode:
                        self.logger.error(f"Error recognizing block: {str(e)}")
                    blk.text = ''
            else:
                if self.debug_mode:
                    self.logger.warning('Invalid text block coordinates for target image')
                blk.text = ''

    def _process_result(self, result):
        try:
            if not result or result[0] is None:
                return ''

            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                result = result[0]

            texts = []
            for line in result:
                if isinstance(line, list) and len(line) > 1 and isinstance(line[1], (list, tuple)) and len(line[1]) > 0:
                    text = line[1][0]
                    text = re.sub(r'-(?!\w)', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    texts.append(text.strip())

            if not texts:
                return ''

            text = ' '.join(texts)
            text = self._apply_no_uppercase(text)
            text = self._apply_punctuation_and_spacing(text)

            return text
        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"Error processing OCR result: {str(e)}")
            return ''

    def _apply_no_uppercase(self, text: str) -> str:
        def process_sentence(sentence):
            words = sentence.split()
            if not words:
                return ''
            return ' '.join([words[0].capitalize()] + [word.lower() for word in words[1:]])

        sentences = re.split(r'(?<=[.!?…])\s+', text)
        return ' '.join(process_sentence(sentence) for sentence in sentences)

    def _apply_punctuation_and_spacing(self, text: str) -> str:
        text = re.sub(r'\s+([,.!?…])', r'\1', text)
        text = re.sub(r'([,.!?…])(?!\s)(?![,.!?…])', r'\1 ', text)
        text = re.sub(r'([,.!?…])\s+([,.!?…])', r'\1\2', text)
        return text.strip()

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ['language', 'device', 'use_angle_cls', 'ocr_version', 'enable_mkldnn', 'det_limit_side_len', 'rec_batch_num', 'drop_score']:
            self.language = self.params['language']['value']
            self.device = self.params['device']['value']
            self.use_angle_cls = self.params['use_angle_cls']['value']
            self.ocr_version = self.params['ocr_version']['value']
            self.enable_mkldnn = self.params['enable_mkldnn']['value']
            self.det_limit_side_len = self.params['det_limit_side_len']['value']
            self.rec_batch_num = self.params['rec_batch_num']['value']
            self.drop_score = self.params['drop_score']['value']
            self._load_model()
        elif param_key == 'text_case':
            self.text_case = self.params['text_case']['value']
        elif param_key == 'output_format':
            self.output_format = self.params['output_format']['value']
