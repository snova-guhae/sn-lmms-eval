from io import BytesIO
import base64
from typing import List, Tuple
from tqdm import tqdm
import time

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from accelerate import Accelerator, DistributedType
from together import Together
from PIL import Image

NUM_SECONDS_TO_SLEEP = 30
from loguru import logger as eval_logger


@register_model("mmllama_together")
class MMLlamaTogether(lmms):
    def __init__(
        self,
        endpoint_url: str = "https://api.together.xyz/v1/chat/completions",
        endpoint_key: str = "",
        timeout: int = 120,
        **kwargs,
    ) -> None:
        super().__init__()

        self.endpoint_url = endpoint_url
        self.endpoint_key = endpoint_key

        self.timeout = timeout
        self.client = Together(api_key=endpoint_key)

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
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            img = self.encode_image(visuals[0])
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if gen_kwargs["max_new_tokens"] > 4096:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 1
            for attempt in range(5):
                try:
                    response = self.client.chat.completions.create(
                        model="meta-llama/Llama-Vision-Free",
                        messages=[{"role": "user", "content": [{"type": "text", "text": contexts}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]}],
                        max_tokens=gen_kwargs["max_new_tokens"],
                        temperature=gen_kwargs["temperature"],
                        stop=["<|eot_id|>", "<|eom_id|>"],
                        stream=False,
                    )
                    response_text = response.choices[0].message.content
                    break  # If successful, break out of the loop

                except Exception as e:
                    try:
                        error_msg = response.json()
                    except:
                        error_msg = ""

                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.\nReponse: {error_msg}")
                    if attempt <= 3:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty string
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.\nResponse: {error_msg}")
                        response_text = ""

            res.append(response_text)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError