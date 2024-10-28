import numpy as np
from typing import List
import easyocr
import os
import cv2

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

# Specify the path for storing EasyOCR models
EASY_OCR_PATH = os.path.join('data', 'models', 'easy-ocr')

@register_OCR('easy_ocr')
class EasyOCRModule(OCRBase):
    # Mapping of language names to EasyOCR language codes
    lang_map = {
        'Abaza': 'abq',
        'Adyghe': 'ady',
        'Afrikaans': 'af',
        'Angika': 'ang',
        'Arabic': 'ar',
        'Assamese': 'as',
        'Avar': 'ava',
        'Azerbaijani': 'az',
        'Belarusian': 'be',
        'Bulgarian': 'bg',
        'Bihari': 'bh',
        'Bhojpuri': 'bho',
        'Bengali': 'bn',
        'Bosnian': 'bs',
        'Simplified Chinese': 'ch_sim',
        'Traditional Chinese': 'ch_tra',
        'Chechen': 'che',
        'Czech': 'cs',
        'Welsh': 'cy',
        'Danish': 'da',
        'Dargwa': 'dar',
        'German': 'de',
        'English': 'en',
        'Spanish': 'es',
        'Estonian': 'et',
        'Persian (Farsi)': 'fa',
        'French': 'fr',
        'Irish': 'ga',
        'Goan Konkani': 'gom',
        'Hindi': 'hi',
        'Croatian': 'hr',
        'Hungarian': 'hu',
        'Indonesian': 'id',
        'Ingush': 'inh',
        'Icelandic': 'is',
        'Italian': 'it',
        'Japanese': 'ja',
        'Kabardian': 'kbd',
        'Kannada': 'kn',
        'Korean': 'ko',
        'Kurdish': 'ku',
        'Latin': 'la',
        'Lak': 'lbe',
        'Lezghian': 'lez',
        'Lithuanian': 'lt',
        'Latvian': 'lv',
        'Magahi': 'mah',
        'Maithili': 'mai',
        'Maori': 'mi',
        'Mongolian': 'mn',
        'Marathi': 'mr',
        'Malay': 'ms',
        'Maltese': 'mt',
        'Nepali': 'ne',
        'Newari': 'new',
        'Dutch': 'nl',
        'Norwegian': 'no',
        'Occitan': 'oc',
        'Pali': 'pi',
        'Polish': 'pl',
        'Portuguese': 'pt',
        'Romanian': 'ro',
        'Russian': 'ru',
        'Serbian (cyrillic)': 'rs_cyrillic',
        'Serbian (latin)': 'rs_latin',
        'Nagpuri': 'sck',
        'Slovak': 'sk',
        'Slovenian': 'sl',
        'Albanian': 'sq',
        'Swedish': 'sv',
        'Swahili': 'sw',
        'Tamil': 'ta',
        'Tabassaran': 'tab',
        'Telugu': 'te',
        'Thai': 'th',
        'Tajik': 'tjk',
        'Tagalog': 'tl',
        'Turkish': 'tr',
        'Uyghur': 'ug',
        'Ukrainian': 'uk',
        'Urdu': 'ur',
        'Uzbek': 'uz',
        'Vietnamese': 'vi',
    }

    params = {
        'language': {
            'type': 'selector',
            'options': list(lang_map.keys()),
            'value': 'English',  # Default language
        },
        'device': DEVICE_SELECTOR(),
        'enable_detection': {
            'type': 'selector',
            'options': ['Enable detection', 'Disable detection'],
            'value': 'Enable detection',
            'description': 'Enable or disable text detection',
        },
        'to_uppercase': {
            'type': 'checkbox',
            'value': False,
            'description': 'Convert text to uppercase',
        },
        'detail': {
            'type': 'checkbox',
            'value': True,
            'description': 'Include information about coordinates in the result',
        },
        'paragraph': {
            'type': 'checkbox',
            'value': True,
            'description': 'Combine results into paragraphs',
        },
        'decoder': {
            'type': 'selector',
            'options': ['greedy', 'beamsearch', 'wordbeamsearch'],
            'value': 'greedy',
            'description': 'Selecting a decoder for text recognition',
        },
        'allowlist': {
            'value': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,!.? ',
            'description': 'Allowed characters for recognition',
        },
        'contrast_ths': {
            'value': 0.1,
            'description': 'Contrast threshold for recognition',
        },
        'adjust_contrast': {
            'value': 0.5,
            'description': 'Contrast adjustment level',
        },
    }
    device = DEFAULT_DEVICE

    def __init__(self, **params):
        super().__init__(**params)
        self.language = self.params['language']['value']
        self.device = self.params['device']['value']
        self.enable_detection = self.params['enable_detection']['value'] == 'Включить детектирование'
        self.to_uppercase = self.params['to_uppercase']['value']
        self.detail = 1 if self.params['detail']['value'] else 0
        self.paragraph = self.params['paragraph']['value']
        self.decoder = self.params['decoder']['value']
        self.allowlist = self.params['allowlist']['value']
        self.contrast_ths = self.params['contrast_ths']['value']
        self.adjust_contrast = self.params['adjust_contrast']['value']
        self.reader = None
        self._load_model()

    def _load_model(self):
        lang_code = self.lang_map[self.language]
        gpu = True if self.device == 'cuda' else False
        if self.debug_mode:
            self.logger.info(f"Загружаем модель для языка: {self.language} ({lang_code}), GPU: {gpu}")
        self.reader = easyocr.Reader(
            lang_list=[lang_code],
            gpu=gpu,
            model_storage_directory=EASY_OCR_PATH,
            download_enabled=True,
            detector=self.enable_detection,
            recognizer=True,
        )

    def ocr_img(self, img: np.ndarray) -> str:
        if self.debug_mode:
            self.logger.debug(f"Начало OCR для изображения размером: {img.shape}")
        if self.enable_detection:
            # Use readtext with original image
            result = self.reader.readtext(
                image=img,
                detail=self.detail,
                paragraph=self.paragraph,
                decoder=self.decoder,
                allowlist=self.allowlist,
                contrast_ths=self.contrast_ths,
                adjust_contrast=self.adjust_contrast
            )
        else:
            # Convert the image to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Image preprocessing (optional)
            _, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            h, w = img_gray.shape
            # Create a bounding box in four-point format
            bbox = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            result = self.reader.recognize(
                img_cv_grey=img_gray,
                horizontal_list=[bbox],
                free_list=[],
                detail=self.detail,
                paragraph=self.paragraph,
                decoder=self.decoder,
                allowlist=self.allowlist,
                contrast_ths=self.contrast_ths,
                adjust_contrast=self.adjust_contrast
            )
        if self.debug_mode:
            self.logger.debug(f"Результат распознавания: {result}")
        text = self._process_result(result)
        return text

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                cropped_img = img[y1:y2, x1:x2]
                if self.debug_mode:
                    self.logger.debug(f"Обработка блока с координатами: ({x1}, {y1}, {x2}, {y2})")
                if self.enable_detection:
                    # Use readtext with cropped image
                    result = self.reader.readtext(
                        image=cropped_img,
                        detail=self.detail,
                        paragraph=self.paragraph,
                        decoder=self.decoder,
                        allowlist=self.allowlist,
                        contrast_ths=self.contrast_ths,
                        adjust_contrast=self.adjust_contrast
                    )
                else:
                    # Convert the image to grayscale
                    cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                    # Image preprocessing (optional)
                    _, cropped_img_gray = cv2.threshold(cropped_img_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    h, w = cropped_img_gray.shape
                    # Create a bounding box in four-point format
                    bbox = np.array([[0, 0], [w, 0], [w, h], [0, h]])
                    result = self.reader.recognize(
                        img_cv_grey=cropped_img_gray,
                        horizontal_list=[bbox],
                        free_list=[],
                        detail=self.detail,
                        paragraph=self.paragraph,
                        decoder=self.decoder,
                        allowlist=self.allowlist,
                        contrast_ths=self.contrast_ths,
                        adjust_contrast=self.adjust_contrast
                    )
                if self.debug_mode:
                    self.logger.debug(f"Результат распознавания блока: {result}")
                text = self._process_result(result)
                blk.text = text
            else:
                if self.debug_mode:
                    self.logger.warning('Некорректные координаты блока текста для целевого изображения')
                blk.text = ''

    def _process_result(self, result):
        if self.detail == 0:
            text = ' '.join(result)
        else:
            # If detail=1, the result is a list with coordinate information
            text = ' '.join([item[1] for item in result])

        if self.to_uppercase:
            text = text.upper()
        return text

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ['language', 'device', 'enable_detection']:
            self.language = self.params['language']['value']
            self.device = self.params['device']['value']
            self.enable_detection = self.params['enable_detection']['value'] == 'Enable detection'
            self._load_model()
        elif param_key == 'to_uppercase':
            self.to_uppercase = self.params['to_uppercase']['value']
        elif param_key == 'detail':
            self.detail = 1 if self.params['detail']['value'] else 0
        elif param_key == 'paragraph':
            self.paragraph = self.params['paragraph']['value']
        elif param_key == 'decoder':
            self.decoder = self.params['decoder']['value']
        elif param_key == 'allowlist':
            self.allowlist = self.params['allowlist']['value']
        elif param_key == 'contrast_ths':
            self.contrast_ths = self.params['contrast_ths']['value']
        elif param_key == 'adjust_contrast':
            self.adjust_contrast = self.params['adjust_contrast']['value']
