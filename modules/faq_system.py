"""
FAQ系统模块，负责存储和匹配问答对
"""
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from utils.embeddings import get_embedding_model, calculate_similarity
from utils.cache import VectorCache
from config import FAQ_SIMILARITY_THRESHOLD_K1, FAQ_SIMILARITY_THRESHOLD_K2
import random
import hashlib
class FAQSystem:
    """FAQ系统类，用于存储和匹配问答对"""
    
    def __init__(self, embeddings: Optional[Embeddings] = None):
        """
        初始化FAQ系统
        
        Args:
            embeddings: 嵌入模型，如果为None则使用默认的智谱AI嵌入模型
        """
        self.embeddings = embeddings or get_embedding_model()
        self.cache = VectorCache(cache_name="faq_cache")
        self.documents = []   
        
    def add_new_qa_to_faq(self, question: str, answer: str, json_file: str) -> None:
        """
        添加问答对到FAQ系统
        
        Args:
            question: 问题
            answer: 答案
        """
        self.documents.append(Document(
            page_content=question,
            metadata={
                'answer': answer,
                'question': question,
                'hash_id': self.get_text_hash(question)
            }
        ))
        with open(json_file, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        faqs.append({'question': question, 'answer': answer})
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(faqs, f, ensure_ascii=False, indent=4)
        
        
    
    def batch_add_faqs(self, faqs: List[Dict[str, str]]) -> None:
        """
        批量添加问答对到FAQ系统
        
        Args:
            faqs: 问答对列表，每个问答对为包含'question'和'answer'键的字典
        """
        self.documents = []
        for faq in faqs:
            question = faq.get('question', '')
            answer = faq.get('answer', '')
            
            if question and answer:
                # 创建文档
                doc = Document(
                    page_content=question,
                    metadata={
                        'answer': answer,
                        'question': question,
                        'hash_id': self.get_text_hash(question)
                    }
                )
                self.documents.append(doc)
    
    def load_faqs_from_json(self, json_file: str) -> None:
        """
        从JSON文件加载问答对
        
        Args:
            json_file: JSON文件路径
        """
        if not os.path.exists(json_file):
            print(f"Warning: FAQ JSON file not found: {json_file}")
            return
        
        with open(json_file, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        
        if isinstance(faqs, list):
            self.batch_add_faqs(faqs)
        else:
            print(f"Warning: Invalid FAQ JSON format in {json_file}")
    
    def search_faq(self, query: str, top_k: int = 1) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        搜索最匹配的FAQ
        
        Args:
            query: 查询问题
            top_k: 返回的最相似问题数量
        
        Returns:
            Tuple[float, Optional[Dict[str, Any]]]: 相似度分数和最匹配的FAQ元数据(如果有)
        """
        if len(self.documents) == 0:
            return 0.0, None
        
        results = None
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=FAQ_SIMILARITY_THRESHOLD_K1,
            )
        results = embeddings_filter.compress_documents(self.documents, query)
        if not results or len(results) == 0:
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=FAQ_SIMILARITY_THRESHOLD_K2,
                )
            results = embeddings_filter.compress_documents(self.documents, query)
        
        if not results or len(results) == 0:
            return 0.0, None
        
        # 获取最相似的问题和分数
        result = random.choice(results)
        
        # 获取查询向量和文档向量
        query_vector = self.embeddings.embed_query(query)
        doc_vector = self.embeddings.embed_query(result.page_content)
        
        # 计算余弦相似度
        similarity = calculate_similarity(query_vector, doc_vector)
        
        # 返回相似度分数和FAQ元数据
        return similarity, {
            'question': result.page_content,
            'answer': result.metadata.get('answer', ''),
            'hash_id': result.metadata.get('hash_id', '')
        }
        
    def get_text_hash(self, text: str) -> str:
        """
        获取文本的哈希值
        
        Args:
            text: 输入文本
        
        Returns:
            str: 文本的MD5哈希值
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            query: 用户查询问题
        
        Returns:
            Dict[str, Any]: 处理结果，包含处理类型和相关信息
        """
        # 搜索FAQ
        similarity_score, faq_info = self.search_faq(query)
        
        if similarity_score < FAQ_SIMILARITY_THRESHOLD_K2:
            # 相似度低于K2，直接发送原始Query给Agent
            return {
                'type': 'direct_to_agent',
                'query': query,
                'similarity_score': similarity_score
            }
        elif FAQ_SIMILARITY_THRESHOLD_K2 <= similarity_score <= FAQ_SIMILARITY_THRESHOLD_K1:
            # 相似度在K1和K2之间，发送Query和缓存答案给Agent
            return {
                'type': 'query_with_cache',
                'query': query,
                'cached_question': faq_info.get('question', ''),
                'cached_answer': faq_info.get('answer', ''),
                'similarity_score': similarity_score
            }
        else:
            # 相似度高于K1，直接返回数据库答案给用户
            return {
                'type': 'direct_answer',
                'query': query,
                'answer': faq_info.get('answer', ''),
                'cached_question': faq_info.get('question', ''),
                'similarity_score': similarity_score
            }

