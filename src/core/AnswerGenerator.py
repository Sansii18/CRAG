from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.utils.config import get_config
from src.utils.logger import logger
from typing import Dict


class AnswerGenerator:
    def __init__(self, domain : str = "general"):
        config_nvidia = get_config("nvidia")

        self.llm = ChatOpenAI(
            model = config_nvidia.llm_model_id,
            api_key = config_nvidia.api_key,
            base_url = config_nvidia.base_url,
            temperature=0.2,
            max_completion_tokens=500,
            timeout=140, # in .env -> NVIDIA_LLM_MODEL_ID = google/gemma-4-31b-it takes 90-120 seconds to load , increase the timeout
            max_retries=1
        )
        self.domain = domain

        self.templates = {
            "high_confidence" : """Based on retrieved information , here is the answer :
            Context: {context}
            Answer : {answer}
            CONFIDENCE: HIGH (>80%)
            Sources: {sources}
            Note : This answer is based on relevant information from trusted sources.
            """,

            "medium_confidence": """Based on available information and web search results, here is the answer:

            Context: {context}
            Answer: {answer}
            CONFIDENCE: MEDIUM (50-80%)
            Sources: {sources}
            Note: This answer combines local knowledge and web search results.
            Consult a qualified professional for personalised advice.
            """,

            "low_confidence": """I cannot reliably answer this question with the available information.

            Query: {query}
            Issue: {issue}

            Recommendations:
            1. Consult a qualified professional
            2. Check official and authoritative sources on this topic
            3. Rephrase your question for better results

            Sources Consulted: {sources}
            CONFIDENCE: LOW (<50%)
            Note: The information available is insufficient. Exercise caution and seek expert advice."""
        }

    def generate_answer(
            self, 
            query : str, 
            context : str,
            confidence : float,
            sources : list
    ) -> Dict : 
        try : 
            if confidence > 0.8:
                confidence_level = "high_confidence"
            elif confidence > 0.5:
                confidence_level = "medium_confidence"
            else:
                confidence_level = "low_confidence"
            
            template = self.templates[confidence_level]

            source_text = "\n".join([
                f"- {s.get('title', 'Unknown')} ({s.get('source_type', 'unknown')})"
                for s in sources[:3]
            ])

            if confidence_level == "low_confidence":
                # NO LLM CALL FOR LOW CONFIDENCE — just return the template with recommendations

                formatted_response = template.format(
                    query=query,
                    issue="Insufficient relevant information in knowledge base",
                    sources=source_text
                )

            else:
                generation_prompt = f"""You are a knowledgeable {self.domain} assistant.

                Based on the following context, generate a clear and accurate answer.

                Context: {context}
                Question : {query}
                Answer : 
                """
                response = self.llm.invoke(generation_prompt)

                answer_text = (
                    response.content
                    if hasattr(response, 'content')
                    else str(response)
                )

                formatted_response = template.format(
                    context = context[:500],
                    answer = answer_text,
                    sources = source_text
                )

            logger.info(
                f"Generated {confidence_level} answer | "
                f"domain={self.domain} | query={query[:50]}..."
            )

            return {
                "answer": formatted_response,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": 0.0,
                "confidence_level": "error",
                "sources": []
            }
