"""
RAG项目主入口文件
"""
import os
import json
import argparse
from typing import Dict, Any, Optional, List
from langchain.memory import ConversationBufferMemory

from modules.faq_system import FAQSystem
from modules.model_agent import ModelAgent
from modules.scoring_model import ScoringModel
from utils.cache import setup_langchain_cache

from config import FAQ_SIMILARITY_THRESHOLD_K1, FAQ_SIMILARITY_THRESHOLD_K2, SCORING_THRESHOLD_K3

class RAGSystem:
    """RAG系统类，整合FAQ系统和模型Agent"""
    
    def __init__(self):
        """初始化RAG系统"""
        # 设置LangChain缓存
        setup_langchain_cache(cache_type="sqlite")
        
        # 初始化FAQ系统
        self.faq_system = FAQSystem()
        
        # 初始化模型Agent
        self.model_agent = ModelAgent()
        
        # 初始化打分模型
        self.scoring_model = ScoringModel()
        
        # 初始化对话记忆
        self.memory = ConversationBufferMemory(return_messages=True)
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            query: 用户查询
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        # 先通过FAQ系统处理
        faq_result = self.faq_system.process_query(query)
        
        if faq_result["type"] == "direct_answer":
            # 相似度高于K2，直接返回FAQ答案
            return {
                "source": "faq",
                "query": query,
                "response": faq_result["answer"],
                "cached_question": faq_result["cached_question"],
                "similarity_score": faq_result["similarity_score"]
            }
        
        elif faq_result["type"] == "query_with_cache":
            # 相似度在K1和K2之间，发送Query和缓存答案给Agent
            cached_info = f"参考信息：\n问题：{faq_result['cached_question']}\n答案：{faq_result['cached_answer']}"
            
            # 调用模型Agent并带上缓存答案
            agent_result = self.model_agent.process(query, context=cached_info)
            
            if agent_result.get("should_regenerate", False):
                # 需要重新生成
                agent_result = self._handle_regeneration(query, agent_result["response"])
            
            return {
                "source": "agent_with_cache",
                "query": query,
                "response": agent_result["response"],
                "cached_question": faq_result["cached_question"],
                "similarity_score": faq_result["similarity_score"],
                "evaluation": agent_result.get("evaluation", {}),
                'message': agent_result.get('message', '')
            }
        
        else:
            # 相似度低于K1，直接发送原始Query给Agent
            agent_result = self.model_agent.process(query)
            
            if agent_result.get("should_regenerate", False):
                # 需要重新生成
                agent_result = self._handle_regeneration(query, agent_result["response"])
            
            return {
                "source": "agent",
                "query": query,
                "response": agent_result["response"],
                "evaluation": agent_result.get("evaluation", {}),
                'message': agent_result.get('message', ''),
                "similarity_score": faq_result["similarity_score"],
            }
    
    def _handle_regeneration(self, query: str, response: str) -> Dict[str, Any]:
        """
        处理需要重新生成的情况
        
        Args:
            query: 用户查询
            response: 当前回复
            
        Returns:
            Dict[str, Any]: 重新生成的结果
        """
        # 尝试重新生成
        regenerated = self.model_agent.regenerate(query, response)
        
        # 如果重新生成仍然不满足要求，但已到达重试次数限制，则返回当前最佳结果
        # if regenerated.get("should_regenerate", False) :
        #     regenerated["message"] = "已达到重试次数限制，返回当前最佳结果"
        #     regenerated["should_regenerate"] = False
        
        return regenerated
    
    def load_faq_from_file(self, file_path: str) -> None:
        """
        从文件加载FAQ
        
        Args:
            file_path: FAQ文件路径
        """
        self.faq_system.load_faqs_from_json(file_path)
    
    def add_new_qa_to_faq(self, question: str, answer: str, json_file: str) -> None:
        """
        添加新的问答对到FAQ系统
        
        Args:
            question: 问题
            answer: 答案
            json_file: FAQ文件路径
        """
        self.faq_system.add_new_qa_to_faq(question, answer, json_file)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统")
    parser.add_argument("--query", "-q", type=str, help="用户查询")
    parser.add_argument("--mode", "-m", choices=["cli", "interactive"], default="interactive", help="运行模式")
    parser.add_argument("--faq", default= '/Users/utopia/Documents/毕设/claude/rag_project/data/faq_sample.json',type=str, help="FAQ文件路径")
    parser.add_argument("--add_faq",'-a', action="store_true", help="添加FAQ")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    # 初始化系统
    rag_system = RAGSystem()
    
    # 加载FAQ
    if args.faq and os.path.exists(args.faq):
        print(f"正在加载FAQ文件: {args.faq}")
        rag_system.load_faq_from_file(args.faq)
    
    if args.mode == "cli" and args.query:
        # 命令行模式
        result = rag_system.process(args.query)
        print(f"查询: \n{args.query}")
        print(f"\n回复: \n{result['response']}")
        
        if 'similarity_score' in result:
            print(f"相似度分数: {result['similarity_score']}")
        
        if 'evaluation' in result:
            print(f"质量评估: {result['evaluation']}")
    
    else:
        # 交互模式
        print("RAG系统启动，输入 'exit' 或 'quit' 退出")
        print("系统配置:")
        print(f"- FAQ相似度阈值 K1: {FAQ_SIMILARITY_THRESHOLD_K1}")
        print(f"- FAQ相似度阈值 K2: {FAQ_SIMILARITY_THRESHOLD_K2}")
        print(f"- 打分模型阈值 K3: {SCORING_THRESHOLD_K3}")
        
        while True:
            query = input("\n请输入查询 (exit/quit/q 退出): ")
            
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            if not query.strip():
                continue
            
            # try:
            result = rag_system.process(query)
            print(f"\n回答: {result['response']}")
            
            if args.verbose:
                print(f"\n\n Detailed Result:\n")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            if args.add_faq:
                rag_system.add_new_qa_to_faq(query, result['response'], args.faq)
            # except Exception as e:
            #     print(f"处理出错: {str(e)}")

if __name__ == "__main__":
    main()