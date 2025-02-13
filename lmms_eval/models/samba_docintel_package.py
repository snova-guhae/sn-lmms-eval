from io import BytesIO
import base64
from typing import List, Tuple
from tqdm import tqdm
import requests as url_requests
import time
from pathlib import Path

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from accelerate import Accelerator, DistributedType

from PIL import Image

NUM_SECONDS_TO_SLEEP = 30
from loguru import logger as eval_logger
from docintel import Chains, DocumentIngestion, LayoutDetection, Llama32OCR, MultiVectorTextRetriever
from docintel import create_basic_logger, cache_crops, load_configs, load_crops


@register_model("samba_docintel_package")
class SambaDocIntelPackage(lmms):
    def __init__(
        self,
        config_path="config.yaml",
        **kwargs,
    ) -> None:
        super().__init__()
        self.configs = load_configs(config_path)
        self.ingestor = DocumentIngestion()
        self.layout_detector = LayoutDetection(layout_detection_configs=self.configs["layout_detection"]["doclaynet"])
        self.ocr_engine = Llama32OCR(
            llama_3_2_ocr_configs=self.configs["llama_3_2_ocr"], langchain_chunking_configs=self.configs["langchain_chunking"], llm_configs=self.configs["llm"], additional_processing_configs=self.configs["additional_processing"]
        )

        self.chains = Chains(llm_configs=self.configs["llm"], retrieval_configs=self.configs["retrieval"])
        self.cache_base = "cache"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc = self.task_dict[task][split][doc_id]
            visuals = [doc_to_visual(doc)]
            visuals = self.flatten(visuals)
            doc_id = doc["doc_id"].replace(".pdf", "")

            crops_path = Path(self.cache_base, "document_images", doc_id)
            ocr_out = None
            if not Path(self.cache_base, "chroma_databases", f"{doc_id}.chromadb").is_dir():
                if crops_path.is_dir():
                    crops = load_crops(crops_path)
                else:
                    crops = self.layout_detector.create_crops_doclaynet(visuals)
                    cache_crops(crops, crops_path)
                ocr_out = self.ocr_engine.process(crops)
            ret_setup = MultiVectorTextRetriever(embedding_configs=self.configs["embedding_model"], retrieval_configs=self.configs["retrieval"], cache_base_directory=self.cache_base, identifier=doc_id)

            retriever = ret_setup.initialize_vectorstore(ocr_output=ocr_out)
            bm25_retriever, bm25_tokenizer = ret_setup.initialize_bm25_retriever()
            answer = self.chains.qa_w_images(retriever=retriever, question=contexts, bm25_retriever=bm25_retriever, bm25_tokenizer=bm25_tokenizer)

            res.append(answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
