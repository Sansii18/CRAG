from typing import Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.utils.config import get_config
from src.utils.logger import logger
import json 

class RetrievalEvaluator:
    def __init__(self, domain:str = 'general'):
        config_nvidia = get_config('nvidia')
        config_crag = get_config('crag')
        self.llm = ChatOpenAI(
            model = config_nvidia.llm_model_id,
            api_key = config_nvidia.api_key,
            base_url = config_nvidia.base_url,
            temperature = 0.2
        )

        self.domain = domain
        self.confidence_threshold = config_crag.retrieval_threshold
