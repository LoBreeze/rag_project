"""
模型Agent模块，负责处理复杂查询和工具调用
"""
from typing import Dict, Any, List, Optional, Union
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.chat_models import ChatZhipuAI
from langgraph.graph import END, MessagesState, StateGraph
from modules.tools.db_query import DBQueryTool
from modules.tools.arxiv_query import ArxivQueryTool
from modules.scoring_model import ScoringModel
from config import ZHIPUAI_API_KEY, LLM_MODEL, MAX_PAPERS

# 定义类型

from langchain.agents import initialize_agent

# tools = [CircumferenceTool()]

# # initialize agent with tools
# agent = initialize_agent(
#     agent='chat-conversational-react-description',
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     max_iterations=3,
#     early_stopping_method='generate',
#     memory=conversational_memory
# )

class ModelAgent:
    """模型Agent类，处理复杂查询和工具调用"""
    
    def __init__(self):
        """初始化模型Agent"""
        # 初始化LLM
        self.llm = ChatZhipuAI(
            model=LLM_MODEL,
            temperature=0.3,
            api_key=ZHIPUAI_API_KEY
        )
        
        # 初始化工具
        self.db_tool = DBQueryTool()
        self.arxiv_tool = ArxivQueryTool()
        # 初始化打分模型
        self.scoring_model = ScoringModel()
        # 初始化工具列表
        self.tools = self._initialize_tools()
        # 初始化系统消息
        self.system_message = self._get_system_message()
        # 初始化状态图
        self.graph = self._create_graph()
        # 初始化对话记忆
        self.message_state = MessagesState(messages=[])
    
    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """
        初始化工具列表
        
        Returns:
            List[Dict[str, Any]]: 工具列表
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_database",
                    "description": "在数据库中搜索相关信息，包括文档和知识库",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索查询"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_arxiv",
                    "description": "在Arxiv上搜索学术论文",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索查询"
                            },
                            "max_papers": {
                                "type": "integer",
                                "description": "返回的最大论文数量",
                                "default": MAX_PAPERS
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        执行工具调用
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            
        Returns:
            str: 工具调用结果
        """
        try:
            if tool_name == "search_database":
                query = tool_args.get("query", "")
                docs = self.db_tool.query(query)
                str_list = []
                for index, doc in enumerate(docs):
                    str_list.append(f"{index + 1}文档:\n{doc.page_content}")
                return "\n\n".join(str_list)
            
            elif tool_name == "search_arxiv":
                query = tool_args.get("query", "")
                max_papers = tool_args.get("max_papers", MAX_PAPERS)
                result = self.arxiv_tool.query(query, max_docs=max_papers)
                
                if not result.get("success", False):
                    return f"搜索失败: {result.get('message', '未知错误')}"
                
                papers = result.get("papers", [])
                if not papers:
                    return "未找到相关论文。"
                
                # 格式化结果
                output = [f"找到 {len(papers)} 篇相关论文:"]
                # print(papers[0])
                for i, paper in enumerate(papers, 1):
                    title = paper.get("title", "未知标题")
                    authors = paper.get("authors", "未知作者")
                    summary = paper.get("ai_summary", paper.get("summary", "无摘要"))
                    url = paper.get("url", "")
                    
                    paper_text = f"""### {i}. {title}
作者: {authors}
摘要: {summary}
链接: {url}
"""
                    output.append(paper_text)
                
                return "\n\n".join(output)
            else:
                return f"未知工具: {tool_name}"
                
        except Exception as e:
            print(f"Error executing tool {tool_name}: {e}")
            return f"工具调用出错: {str(e)}"
    
    def _get_system_message(self) -> str:
        """
        获取系统消息
        
        Returns:
            str: 系统消息
        """
        return """你是一个智能AI助手，专注于解答用户问题并提供准确、全面的信息。

在处理问题时，请遵循以下原则：
1. 采用ReACT(Reasoning and Acting)方法，即"观察-思考-行动"的迭代过程
2. 使用Chain-of-Thought(CoT)思维链，通过逐步推理解决问题
3. 在需要时调用可用的工具获取信息
4. 给出准确、有根据的回答，避免虚构信息

可用工具:
- search_database: 在内部数据库中搜索相关信息
- search_arxiv: 在Arxiv上搜索学术论文

当你不确定答案或需要更多信息时，请主动使用这些工具。首先思考需要获取什么信息，然后选择合适的工具进行查询。

请保持响应的专业性和客观性，并在回答中明确引用信息来源。"""
    
    def _run_agent_with_tools(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用工具运行Agent
        
        Args:
            messages: 消息列表
            
        Returns:
            Dict[str, Any]: Agent输出
        """
        # 准备消息
        langchain_messages = []
        
        # 添加系统消息
        langchain_messages.append(SystemMessage(content=self.system_message))
        # 添加对话历史
        langchain_messages.extend(messages)
        
        # 调用LLM
        response = self.llm.invoke(
            langchain_messages,
            tools=self.tools
        )
        
        return response
    
    def _tool_decision(self, message_state) -> MessagesState:
        """
        决定是否使用工具
        
        Args:
            messages: 消息列表
            
        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: 工具调用或最终输出
        """
        # 运行Agent
        messages = message_state["messages"]
        response = self._run_agent_with_tools(messages)
        # 检查是否调用工具
        if response.additional_kwargs.get("tool_calls"):
            tool_calls = response.additional_kwargs["tool_calls"]
            message_state["messages"].append(AIMessage(content=response.content, additional_kwargs=tool_calls[0]))
        else:
            message_state["messages"].append(AIMessage(content=response.content))
        # 无工具调用，返回最终输出
        return message_state
    
    def _call_tool(self, message_state: MessagesState) -> List[Dict[str, Any]]:
        """
        调用工具并返回结果
        
        Args:
            messages: 消息列表
            tool_calls: 工具调用列表
            
        Returns:
            List[Dict[str, Any]]: 更新后的消息列表
        """
        # 添加AI的工具调用消息
        last_message = message_state["messages"][-1]
        tool_calls = last_message.additional_kwargs
        # print(tool_calls)
        # 添加工具调用信息
        # 处理每个工具调用
        function_info = tool_calls['function']
        tool_name = function_info.get("name", "")
        tool_call_id = tool_calls.get("id", "")
        try:
            # 解析参数
            args_str = function_info.get("arguments", "{}")
            args = json.loads(args_str)
            
            # 执行工具调用
            result = self.execute_tool(tool_name, args)
            
            # 添加工具响应消息
            message_state["messages"].append(ToolMessage(content=result, tool_call_id=tool_call_id))
            
        except Exception as e:
            # 工具调用失败
            message_state["messages"].append(ToolMessage(content=e, tool_call_id=tool_call_id))
        return message_state
    
    def _check_if_done(self, message_state) -> Union[str, Dict[str, Any]]:
        """
        检查是否需要继续执行工具调用
        
        Args:
            messages: 消息列表
            
        Returns:
            Union[str, Dict[str, Any]]: "continue"或最终状态
        """
        last_message = message_state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return "continue"
        if last_message.additional_kwargs:
            return "tool_call"
        return END
    
    def _create_graph(self) -> StateGraph:
        """
        创建状态图
        
        Returns:
            StateGraph: 状态图
        """
        from langgraph.graph import StateGraph
        
        # 创建状态图
        workflow = StateGraph(MessagesState)
        
        # 添加节点
        workflow.add_node("agent", self._tool_decision)
        workflow.add_node("action", self._call_tool)
        
        # 添加边
        workflow.add_edge("action", "agent")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "agent",
            self._check_if_done,
            {
                "continue": "agent",
                "tool_call": 'action',
                END:END
            }
        )
        
        # 设置入口节点
        workflow.set_entry_point("agent")
        
        # 编译状态图
        return workflow.compile()
    
    def process(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            query: 用户查询
            context: 可选的上下文信息
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        # 准备初始消息
        

        # 添加上下文信息（如果有）
        if context:
            context_message = f"相关上下文信息：\n{context}\n\n用户查询：{query}"
            self.message_state["messages"].append(HumanMessage(content=context_message))
        else:
            self.message_state["messages"].append(HumanMessage(content=query))
        
        # 执行状态图
        result = self.graph.invoke(self.message_state)
        final_messages = result.get("messages", [])[-1]
        
        # 提取最终回复
        ai_messages = final_messages if isinstance(final_messages, AIMessage) else None
        if not ai_messages:
            return {
                "success": False,
                "message": "无法生成回复",
                "response": ""
            }
        
        final_response = ai_messages.content
        
        # 评估回复质量
        should_regen, evaluation = self.scoring_model.should_regenerate(query, final_response)
        self.message_state["messages"].append(AIMessage(content=final_response))
        if should_regen:
            # 分数低于阈值，需要重新生成
            return {
                "success": False,
                "message": "回复质量不满足要求",
                "response": final_response,
                "evaluation": evaluation,
                "should_regenerate": True
            }
        
        # 回复质量符合要求
        return {
            "success": True,
            "message": "生成成功",
            "response": final_response,
            "evaluation": evaluation,
            "should_regenerate": False
        }
    
    def regenerate(self, query: str, previous_response: str) -> Dict[str, Any]:
        """
        重新生成回复
        
        Args:
            query: 用户查询
            previous_response: 之前的回复
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        # 准备包含之前回复的消息
        message_state = MessagesState(messages=[])
        message_state["messages"].append(HumanMessage(content=query))
        message_state["messages"].append(AIMessage(content=previous_response))
        message_state["messages"].append(HumanMessage(content="请根据我的问题重新生成一个更好的回答，上面的回答不够好，需要改进。"))
        
        
        # 执行状态图
        result = self.graph.invoke(message_state)
        final_messages = result.get("messages", [])[-1]
        
        # 提取最终回复
        ai_messages = final_messages if isinstance(final_messages, AIMessage) else None
        if not ai_messages:
            return {
                "success": False,
                "message": "无法生成回复",
                "response": ""
            }
        
        final_response = ai_messages.content
        
        # 评估回复质量
        should_regen, evaluation = self.scoring_model.should_regenerate(query, final_response)

        if should_regen:
            # 分数低于阈值，仍然需要重新生成，但已达到重试限制
            return {
                "success": True,  # 强制返回，不再重试
                "message": "回复质量仍不理想，但已达到重试限制",
                "response": final_response,
                "evaluation": evaluation,
                "should_regenerate": False
            }
        
        # 回复质量符合要求
        return {
            "success": True,
            "message": "重新生成成功",
            "response": final_response,
            "evaluation": evaluation,
            "should_regenerate": False
        }
        
        
if __name__ == '__main__':
    # 测试模型Agent
    agent = ModelAgent()
    
    query = "在arxiv上搜索：blip2"
    result = agent.process(query)
    print(result)
    
    print("\n\n\n\n\n\n")
    query = query
    response = result
    result = agent.regenerate(query, response['response'])
    print(result)