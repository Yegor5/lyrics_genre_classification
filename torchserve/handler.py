import json
import torch
import logging
import os

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from ts.torch_handler.base_handler import BaseHandler
from predict.predict import PredictionProcessor
from safetensors.torch import load_file


logger = logging.getLogger(__name__)


class GenreHandler(BaseHandler):

    def initialize(self, ctx):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = ctx.system_properties["model_dir"]
        ser_file = ctx.manifest["model"]["serializedFile"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            state_dict=load_file(os.path.join(model_dir, ser_file))
        )
        self.model.to(self.device)
        self.model.eval()

        with open(f"{model_dir}/label_names.json") as f:
            label_names = json.load(f)

        self.processor = PredictionProcessor(
            model=self.model,
            tokenizer=self.tokenizer,
            label_names=label_names,
            threshold=0.5
        )

        self.max_length = 512
        logger.info("Model and processor initialized")

    def preprocess(self, data):
        texts = []

        for row in data:
            body = row["body"]
            if isinstance(body, (bytes, bytearray)):
                body = body.decode("utf-8")

            obj = json.loads(body)
            texts.append(obj["text"])

        return texts

    def inference(self, texts):
        return self.processor.predict_batch(
            texts=texts,
            max_length=self.max_length
        )

    def postprocess(self, outputs):
        return outputs