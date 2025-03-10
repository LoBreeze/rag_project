"""
Arxiv查询工具，用于获取学术论文信息
"""
from typing import List, Dict, Any, Optional
from langchain_community.retrievers import ArxivRetriever
from langchain.schema.document import Document
from langchain_community.chat_models import ChatZhipuAI

from config import ZHIPUAI_API_KEY, LLM_MODEL

class ArxivQueryTool:
    """Arxiv查询工具类"""
    
    def __init__(self):
        """初始化Arxiv查询工具"""
        self.retriever = ArxivRetriever(
            load_max_docs=5,
            load_all_available_meta=True,
            doc_content_chars_max=40000
        )
        self.llm = ChatZhipuAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=ZHIPUAI_API_KEY
        )
    
    def search(self, query: str, max_docs: int = 3) -> List[Document]:
        """
        搜索Arxiv论文
        
        Args:
            query: 查询字符串
            max_docs: 返回的最大文档数量
            
        Returns:
            List[Document]: 检索到的文档列表
        """
        try:
            self.retriever.load_max_docs = max_docs
            docs = self.retriever.invoke(query)
            return docs
        except Exception as e:
            print(f"Error searching Arxiv: {e}")
            return []
    
    def format_paper_info(self, doc: Document) -> Dict[str, Any]:
        """
        格式化论文信息
        
        Args:
            doc: 论文文档
            
        Returns:
            Dict[str, Any]: 格式化后的论文信息
        """
        metadata = doc.metadata
        return {
            'title': metadata.get('Title', 'Unknown Title'),
            'authors': metadata.get('Authors', 'Unknown Authors'),
            'summary': metadata.get('Summary', 'No summary available'),
            'published': metadata.get('Published', 'Unknown date'),
            'url': metadata.get('Entry_ID', ''),
            'pdf_url': metadata.get('pdf_url', '')
        }
    
    def summarize_paper(self, doc: Document) -> str:
        """
        使用LLM对论文进行摘要
        
        Args:
            doc: 论文文档
            
        Returns:
            str: 论文摘要
        """
        paper_info = self.format_paper_info(doc)
        content = doc.page_content[:10000]  # 截取部分内容以避免超出token限制
        
        prompt = f"""请对以下学术论文进行简明扼要的总结，突出其主要贡献、方法和结论。
        
论文标题: {paper_info['title']}
论文作者: {paper_info['authors']}
发表日期: {paper_info['published']}

论文摘要:
{paper_info['summary']}

论文内容片段:
{content}

请用中文提供300字左右的总结，重点关注论文的创新点、方法和结论。"""
        
        try:
            result = self.llm.invoke(prompt).content
            return result
        except Exception as e:
            print(f"Error summarizing paper: {e}")
            return f"无法生成摘要。原始摘要: {paper_info['summary']}"
    
    def query(self, query: str, max_docs: int = 3, summarize: bool = True) -> Dict[str, Any]:
        """
        查询Arxiv并返回结果
        
        Args:
            query: 查询字符串
            max_docs: 返回的最大文档数量
            summarize: 是否生成摘要
            
        Returns:
            Dict[str, Any]: 查询结果
        """
        docs = self.search(query, max_docs)
        
        if not docs:
            return {
                'success': False,
                'message': '未找到相关论文',
                'papers': []
            }
        
        papers = []
        for doc in docs:
            paper_info = self.format_paper_info(doc)
            
            if summarize:
                paper_info['ai_summary'] = self.summarize_paper(doc)
            
            papers.append(paper_info)
        
        return {
            'success': True,
            'message': f'找到 {len(papers)} 篇相关论文',
            'papers': papers
        }