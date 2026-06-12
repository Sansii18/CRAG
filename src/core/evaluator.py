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

        self.evaluation_prompt = {
            "general" : """You are an expert evaluator. Assess if the retrieved documents adequately answer the user's query.
            Query : {query}
            Retrieved Documents : {retrieved_docs}
            Evaluate : 
            1. Does the content directly address the query? (Yes/No)
            2. Is the information accurate and relevant? (Yes/No)
            3. Confidence Score (0.0 - 1.0) : ?

            Respond only in JSON format as follows: 
            {{
                "address_query": true/false,
                "is_relevant": true/false,
                "confidence_score": float (0.0 - 1.0),
                "reasoning": "brief explanation"
            }}""",

            "legal" : """"
            You are a legal expert evaluator. Assess if the retrieved documents adequately answer the user's legal query.

            Query : {query}
            Retrieved Documents : {retrieved_docs}  

            Evaluate :
            1. Does the content address the legal question? (Yes/No)
            2. Is the information relevant and accurate? (Yes/No)
            3. Confidence Score (0.0-1.0): ?

            Respond only in JSON format as follows: 
            {{
                "address_query": true/false,
                "is_relevant": true/false,
                "confidence_score": float (0.0 - 1.0),
                "reasoning": "brief explanation"
            }}""",

            "technical" : """"
            You are a technical expert evaluator. Assess if the retrieved documents adequately answer the user's technical query.

            Query : {query}
            Retrieved Documents : {retrieved_docs}

            Evaluate :
            1. Does the content answer the technical question? (Yes/No)
            2. Is the information relevant and accurate? (Yes/No)
            3. Confidence Score (0.0-1.0): ?

            Respond only in JSON format as follows: 
            {{
                "address_query": true/false,
                "is_relevant": true/false,
                "confidence_score": float (0.0 - 1.0),
                "reasoning": "brief explanation"
            }}""",

            "financial" : """"
            You are a financial expert evaluator. Assess if the retrieved documents adequately answer the user's financial query.

            Query : {query}
            Retrieved Documents : {retrieved_docs}

            Evaluate :
            1. Does the content address the financial question? (Yes/No)
            2. Is the information relevant and accurate? (Yes/No)
            3. Confidence Score (0.0-1.0): ?

            Respond only in JSON format as follows: 
            {{
                "address_query": true/false,
                "is_relevant": true/false,
                "confidence_score": float (0.0 - 1.0),
                "reasoning": "brief explanation"
            }}"""
        }

    def evaluate(self , query : str, retrived_docs: list) -> Tuple[float, Dict]:
        """Evaluate retrieved documents and return confidence score."""

        try:
            doc_text = "\n---\n".join([
                f"Document {i+1}: {doc['text'][:500]}"
                for i, doc in enumerate(retrived_docs)
            ])

            prompt_template_str = self.evaluation_prompt.get(
                self.domain,
                self.evaluation_prompt["general"] 
            )

            prompt = PromptTemplate(
                input_variables = ["query", "retrieved_docs"],
                template = prompt_template_str
            )

            formatted_prompt = prompt.format(
                query=query,
                retrieved_docs=doc_text
            )

            response = self.llm.invoke(formatted_prompt)

            response_text = response.content if hasattr(response, 'content') else str(response)

            evaluation = json.loads(response_text)
            confidence = evaluation.get("confidence_score", 0.0)

            logger.info(
                f"Evaluation | domain={self.domain} | "
                f"confidence={confidence:.3f} | query={query[:50]}..."
            )

            return confidence, evaluation
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}. Using default.")
            return 0.5, {"confidence_score": 0.5, "reasoning": "JSON parsing error"}
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.3, {"confidence_score": 0.3, "reasoning": str(e)}
        
    def get_action(self, confidence: float) -> str:
        """Determine action based on confidence score."""
        if confidence > self.confidence_threshold:
            return "GENERATE_ANSWER"
        elif confidence > self.confidence_threshold * 0.75:
            return "WEB_SEARCH"
        else:
            return "REFUSE"
