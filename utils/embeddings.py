"""
嵌入模型工具，封装智谱AI的Embedding模型
"""
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import numpy as np
from config import ZHIPUAI_API_KEY, EMBEDDING_MODEL, FAISS_CHUNK_SIZE, FAISS_CHUNK_OVERLAP

def get_embedding_model():
    """
    获取智谱AI的Embedding模型
    
    Returns:
        ZhipuAIEmbeddings: 嵌入模型实例
    """
    return ZhipuAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=ZHIPUAI_API_KEY
    )

def get_text_splitter():
    """
    获取文本分割器
    
    Returns:
        RecursiveCharacterTextSplitter: 文本分割器实例
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=FAISS_CHUNK_SIZE,
        chunk_overlap=FAISS_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个嵌入向量之间的余弦相似度
    
    Args:
        embedding1: 第一个嵌入向量
        embedding2: 第二个嵌入向量
    
    Returns:
        float: 余弦相似度，范围[-1, 1]，值越大表示越相似
    """

        # 点积
    dot_product = np.dot(vec1, vec2)
    # 计算模长
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # 返回余弦相似度
    return dot_product / (norm_vec1 * norm_vec2)


def batch_embed_texts(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """
    批量获取文本的嵌入向量
    
    Args:
        texts: 文本列表
        batch_size: 批处理大小
    
    Returns:
        List[List[float]]: 嵌入向量列表
    """
    embeddings = []
    embedding_model = get_embedding_model()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    
    return embeddings