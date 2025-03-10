"""
缓存机制，用于缓存FAQ和嵌入向量
"""
import os
import json
import pickle
from typing import Dict, List, Any, Optional
import hashlib
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache
import numpy as np
from config import CACHE_DIR

class VectorCache:
    """向量缓存类，用于缓存和检索嵌入向量"""
    
    def __init__(self, cache_name: str = "embeddings"):
        """
        初始化向量缓存
        
        Args:
            cache_name: 缓存名称
        """
        self.cache_dir = os.path.join(CACHE_DIR, cache_name)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.vector_file = os.path.join(self.cache_dir, "vectors.pkl")
        self.metadata_file = os.path.join(self.cache_dir, "metadata.json")
        
        # 初始化或加载缓存
        if os.path.exists(self.vector_file) and os.path.exists(self.metadata_file):
            self.load_cache()
        else:
            self.vectors = {}  # 文本哈希 -> 向量
            self.metadata = {}  # 文本哈希 -> 元数据
            self.save_cache()
    
    def get_text_hash(self, text: str) -> str:
        """
        获取文本的哈希值
        
        Args:
            text: 输入文本
        
        Returns:
            str: 文本的MD5哈希值
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def add_vector(self, text: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加向量到缓存
        
        Args:
            text: 输入文本
            vector: 嵌入向量
            metadata: 与向量相关的元数据
        """
        text_hash = self.get_text_hash(text)
        self.vectors[text_hash] = vector
        
        if metadata is None:
            metadata = {}
        
        # 添加原始文本到元数据
        metadata['text'] = text
        
        self.metadata[text_hash] = metadata
        self.save_cache()
    
    def get_vector(self, text: str) -> Optional[List[float]]:
        """
        从缓存获取向量
        
        Args:
            text: 输入文本
        
        Returns:
            Optional[List[float]]: 如果缓存中存在，则返回向量，否则返回None
        """
        text_hash = self.get_text_hash(text)
        return self.vectors.get(text_hash)
    
    def get_metadata(self, text: str) -> Optional[Dict[str, Any]]:
        """
        从缓存获取元数据
        
        Args:
            text: 输入文本
        
        Returns:
            Optional[Dict[str, Any]]: 如果缓存中存在，则返回元数据，否则返回None
        """
        text_hash = self.get_text_hash(text)
        return self.metadata.get(text_hash)
    
    def save_cache(self) -> None:
        """保存缓存到磁盘"""
        with open(self.vector_file, 'wb') as f:
            pickle.dump(self.vectors, f)
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def load_cache(self) -> None:
        """从磁盘加载缓存"""
        with open(self.vector_file, 'rb') as f:
            self.vectors = pickle.load(f)
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self.vectors = {}
        self.metadata = {}
        self.save_cache()
    
    def get_all_vectors(self) -> Dict[str, List[float]]:
        """
        获取所有缓存的向量
        
        Returns:
            Dict[str, List[float]]: 文本哈希 -> 向量的字典
        """
        return self.vectors
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有缓存的元数据
        
        Returns:
            Dict[str, Dict[str, Any]]: 文本哈希 -> 元数据的字典
        """
        return self.metadata

def setup_langchain_cache(cache_type: str = "sqlite") -> None:
    """
    设置LangChain缓存
    
    Args:
        cache_type: 缓存类型，'sqlite'或'memory'
    """
    if cache_type == "sqlite":
        cache_file = os.path.join(CACHE_DIR, "langchain.db")
        set_llm_cache(SQLiteCache(database_path=cache_file))
    else:
        set_llm_cache(InMemoryCache())

if __name__ == '__main__':
    cache = VectorCache()
    cache.add_vector("Hello, world!", [0.1, 0.2, 0.3], {"source": "test"})
    print(cache.get_vector("Hello, world!"))
    print(cache.get_metadata("Hello, world!"))
    print(cache.get_all_vectors())
    print(cache.get_all_metadata())
    cache.clear_cache()
    print(cache.get_all_vectors())
    print(cache.get_all_metadata())
    setup_langchain_cache('memory')