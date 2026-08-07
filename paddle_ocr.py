from paddleocr import PaddleOCR
import time

# 默认使用 PP-OCRv6 模型
ocr = PaddleOCR(
    use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
    use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
    lang="ch",
    device="gpu",
    engine="paddle_static",
)
# ocr = PaddleOCR(lang="en") # 通过 lang 参数来使用英文模型
# ocr = PaddleOCR(ocr_version="PP-OCRv5") # 通过 ocr_version 参数切换为 PP-OCRv5 版本
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # 通过 ocr_version 参数切换为 PP-OCRv4 版本
# ocr = PaddleOCR(device="gpu") # 通过 device 参数使得在模型推理时使用 GPU
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_server_det",
#     text_recognition_model_name="PP-OCRv5_server_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# ) # 使用 PP-OCRv5 的 server 模型
start = time.time()
result = ocr.predict("./general_ocr_002.png")
print(f"result: {len(result)}")
print(f"cost: {time.time() - start:.2f}s")
text_result = ""
for res in result:
    # res.print()
    # res.save_to_img("output_ocr")
    # res.save_to_json("output_ocr")
    # print(f"res: {type(res)}")
    print(f"dir: {type(res)}")

    for k, v in res.items():
        if k == "rec_texts":
            text_result += str(v)

print(f"result: {text_result}")