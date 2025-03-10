"""
打分模型，评估Agent回复的质量
"""
from typing import Dict, Any, Tuple

from langchain_community.chat_models import ChatZhipuAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from config import ZHIPUAI_API_KEY, LLM_MODEL, SCORING_THRESHOLD_K3

class ScoringResult(BaseModel):
    """打分结果模型"""
    score: float = Field(description="回复质量得分，范围从0到1")
    reasoning: str = Field(description="打分理由")
    suggestions: str = Field(description="改进建议")

class ScoringModel:
    """打分模型类，评估Agent回复的质量"""
    
    def __init__(self):
        """初始化打分模型"""
        self.llm = ChatZhipuAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=ZHIPUAI_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=ScoringResult)
    
    def evaluate_response(self, query: str, response: str) -> Tuple[float, Dict[str, Any]]:
        """
        评估Agent回复的质量
        
        Args:
            query: 用户查询
            response: Agent的回复
            
        Returns:
            Tuple[float, Dict[str, Any]]: 得分和详细评估结果
        """
        # 创建评估提示
        prompt_template = PromptTemplate(
            template="""请评估以下AI助手回复的质量。你需要从相关性、准确性、完整性、清晰度和有用性等方面进行评估。

用户查询:
{query}

AI助手回复:
{response}

请对回复进行评分，并给出理由和改进建议。你的评分应该在0到1之间，其中1表示完美回复，0表示完全不合格的回复。

{format_instructions}""",
            input_variables=["query", "response"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        # 生成提示
        prompt = prompt_template.format(
            query=query,
            response=response
        )
        # print(prompt)
        try:
            # 获取评估结果
            result = self.llm.invoke(prompt).content
            
            # 解析结果
            parsed_result = self.parser.parse(result)
            
            return parsed_result.score, {
                'score': parsed_result.score,
                'reasoning': parsed_result.reasoning,
                'suggestions': parsed_result.suggestions
            }
        except Exception as e:
            print(f"Error evaluating response: {e}")
            # 出错时返回一个默认值
            return 0.0, {
                'score': 0.0,
                'reasoning': f"评估过程出错: {str(e)}",
                'suggestions': "无法提供建议，因为评估过程失败。"
            }
    
    def should_regenerate(self, query: str, response: str) -> Tuple[bool, Dict[str, Any]]:
        """
        判断是否需要重新生成回复
        
        Args:
            query: 用户查询
            response: Agent的回复
            
        Returns:
            Tuple[bool, Dict[str, Any]]: 是否需要重新生成和评估结果
        """
        score, evaluation = self.evaluate_response(query, response)
        
        # 判断分数是否低于阈值
        should_regen = score < SCORING_THRESHOLD_K3
        
        return should_regen, evaluation
    
if __name__ == "__main__":
    # 测试打分模型
    model = ScoringModel()
    
    query = "如何注册微信小程序？"
    response = "你可以在微信公众平台上注册小程序。"
    
    score, evaluation = model.evaluate_response(query, response)
    print(f"Score: {score}")
    print(f"Evaluation: {evaluation}")
    
    should_regen, evaluation = model.should_regenerate(query, response)
    print(f"Should regenerate: {should_regen}")
    print(f"Evaluation: {evaluation}")