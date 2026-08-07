from llama_index.core import Document
from image_ocr.image_ocr_loader import ImageOCRLoader
import time

loader = ImageOCRLoader(use_gpu=True)
start = time.time()
documents = loader.load_data("./general_ocr_002.png")
print(f"cost: {time.time() - start:.2f}s")
print(documents)