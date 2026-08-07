import os

from llama_index.core.readers.base import BaseReader
from paddleocr import PaddleOCR
from llama_index.core.schema import Document
from typing import Union
import asyncio


class ImageOCRLoader(BaseReader):
    """
    使用PP-OCRv6从图像中提取文本并返回Document
    lang: 语言
    use_gpu: 是否使用GPU
    sem: 并发量
    """
    def __init__(self, lang: str = "ch", use_gpu: bool = False, sem: int = 5, **kwargs):
        # 默认使用 PP-OCRv6 模型
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
            use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
            use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
            lang=lang,
            device="gpu" if use_gpu else "cpu",
            engine="paddle_static",
            **kwargs,
        )
        self.sem = sem

    def load_data(self, file: Union[str, list[str]]) -> list[Document]:
        """
        从单个或多个图像中提取文本，返回Document列表
        file: 单个图像路径或图像路径列表
        Returns:
            list[Document]: Document列表
        """
        return asyncio.run(self.aload_data(file))

    async def aload_data(self, file: Union[str, list[str]]) -> list[Document]:
        """
        从单个或多个图像中提取文本，返回Document列表
        file: 单个图像路径或图像路径列表
        Returns:
            list[Document]: Document列表
        """
        if isinstance(file, str):
            file = [file]
        for file_path in file:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File path {file_path} not found")

        return await self.batch_load_images(file)

    async def load_data_single_image(self, file: str, sem: asyncio.Semaphore) -> Document:
        """
        从单个图像中提取文本，返回Document
        file: 图像路径
        Returns:
            Document: Document
        """
        async with sem:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File path {file} not found")

            # 未来再支持远端url
            # if file.startswith("http"):
            #     async with httpx.AsyncClient() as client:
            #         response = await client.get(file)
            #         image_bytes = response.content
            # else:
            #     with open(file, "rb") as f:
            #         image_bytes = f.read()

            # predict 是CPU密集型同步操作，直接在协程里调用会卡住整个事件循环，让其他协程无法运行，整个信号量也失去意义。需要改为用event_loop
            # ocr_result = self.ocr.predict(file)
            loop = asyncio.get_running_loop()
            ocr_result = await loop.run_in_executor(None, self.ocr.predict, file)
            text_result = ""
            for res in ocr_result:
                for k, v in res.items():
                    if k == "rec_texts":
                        text_result += "\n".join(v)

            return Document(text=text_result, metadata={"source": file})

    async def batch_load_images(self, files: list[str]) -> list[Document]:
        """
        批量加载图像，返回Document列表
        """
        real_sem = asyncio.Semaphore(self.sem)
        tasks = [
            self.load_data_single_image(file, real_sem) for file in files
        ]
        documents = []
        for sub_task in asyncio.as_completed(tasks):
            result = await sub_task
            documents.append(result)

        return documents