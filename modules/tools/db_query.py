"""
数据库查询工具，包括Elasticsearch和FAISS查询
实现了文档检索、重排序和上下文压缩
"""
import os
from typing import List
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_elasticsearch import ElasticsearchStore, BM25Strategy
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, DirectoryLoader
from langchain.retrievers.document_compressors import LLMChainExtractor,LLMListwiseRerank, EmbeddingsFilter
from langchain.schema.document import Document
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_transformers import LongContextReorder
import faiss
from langchain_openai import ChatOpenAI
from utils.embeddings import get_embedding_model, get_text_splitter
from config import (
    CSV_DIR, PDF_DIR, ELASTICSEARCH_URL, ELASTICSEARCH_INDEX,
    FAISS_INDEX_NAME, MAX_DOCUMENTS_AFTER_RERANK, ZHIPUAI_API_KEY, LLM_MODEL,
    ELASTICSEARCH_RECREATE, OPEN_AI_API_KEY, OPEN_AI_MODEL, USE_LLM_FOR_RERANK, USE_LLM_FOR_COMPRESSION, USE_EMBEDDING_FOR_COMPRESSION_THRESHOLD
)

class DBQueryTool:
    """数据库查询工具，集成Elasticsearch和FAISS"""
    
    def __init__(self):
        """初始化数据库查询工具"""
        self.embedding_model = get_embedding_model()
        self.text_splitter = get_text_splitter()
        self.llm = ChatZhipuAI(
            model=LLM_MODEL,
            temperature=0,
            zhipuai_api_key=ZHIPUAI_API_KEY
        )
        self.elastic_store = None
        self.faiss_store = None
        self.reranker = None
        self.compressor = None
        
    
    def initialize_elastic(self, force_reload: bool = False) -> None:
        """
        初始化Elasticsearch
        
        Args:
            force_reload: 是否强制重新加载数据
        """
        # 检查Elasticsearch是否已经存在该索引
        if self.elastic_store is not None and not force_reload:
            return
        
        try:
            # 加载CSV文件
            csv_loader = DirectoryLoader(
                CSV_DIR,
                glob="**/*.csv",
                loader_cls=CSVLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            csv_docs = csv_loader.load()
            
            if not csv_docs:
                print("No CSV documents found, Elasticsearch will not be initialized.")
                return
            
            
            
            # 分割文档
            # split_docs = self.text_splitter.split_documents(csv_docs)
            split_docs = []
            for doc in csv_docs:
                split_docs.append(Document(page_content=doc.page_content, metadata=self._parse_metadata(doc.page_content)))
            
            # 初始化Elasticsearch
            self.elastic_store = ElasticsearchStore(
                es_url=ELASTICSEARCH_URL,
                index_name=ELASTICSEARCH_INDEX,
                embedding=self.embedding_model,
                strategy=BM25Strategy()
            )
            
            if ELASTICSEARCH_RECREATE:
                self.elastic_store.client.indices.delete(
                    index=ELASTICSEARCH_INDEX,
                    ignore_unavailable=True,
                    allow_no_indices=True,
                )
            
            # 添加文档到Elasticsearch
            self.elastic_store.add_documents(split_docs)
            print(f"Loaded {len(split_docs)} documents into Elasticsearch.")
            
        except Exception as e:
            print(f"Error initializing Elasticsearch: {e}")
            self.elastic_store = None
    
    def initialize_faiss(self, force_reload: bool = False) -> None:
        """
        初始化FAISS
        
        Args:
            force_reload: 是否强制重新加载数据
        """
        faiss_index_path = f"{PDF_DIR}/{FAISS_INDEX_NAME}"
        # 如果已经加载并且不需要强制重新加载，则直接返回
        if self.faiss_store is not None and not force_reload:
            return
        
        # 如果存在FAISS索引并且不需要强制重新加载，则从磁盘加载
        if os.path.exists(faiss_index_path) and not force_reload:
            try:
                self.faiss_store = FAISS.load_local(
                    faiss_index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded FAISS index from {faiss_index_path}")
                return
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
        
        # 否则，重新加载PDF文件并创建索引
        try:
            # 使用DirectoryLoader加载所有PDF文件
            pdf_loader = DirectoryLoader(
                PDF_DIR,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            
            if not pdf_docs:
                print("No PDF documents found, FAISS will not be initialized.")
                return
            
            # 分割文档
            split_docs = self.text_splitter.split_documents(pdf_docs)
            # 初始化
            index = faiss.IndexFlatL2(len(self.embedding_model.embed_query("hello world")))
            self.faiss_store = FAISS(
                embedding_function=self.embedding_model,
                index=index,
                docstore={},
                index_to_docstore_id={},
            )
            
            if len(split_docs) < 64:
            # 创建FAISS索引
                self.faiss_store = FAISS.from_documents(
                    split_docs,
                    self.embedding_model
                )
            else:
                batch_size = 50
                # 新增：分批处理逻辑（网页3最佳实践）
                self.faiss_store = FAISS.from_documents(
                    documents=split_docs[:batch_size],  # 初始批次创建索引
                    embedding=self.embedding_model
                )
                # 剩余文档分批次添加（网页5示例）
                for i in tqdm(range(batch_size, len(split_docs), batch_size)):
                    batch = split_docs[i:i + batch_size]
                    self.faiss_store.add_documents(batch)
            
            # 保存FAISS索引
            os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
            self.faiss_store.save_local(faiss_index_path)
            print(f"Created and saved FAISS index with {len(split_docs)} documents.")
            
        except Exception as e:
            print(f"Error initializing FAISS: {e}")
            self.faiss_store = None
    
    def query_elastic(self, query: str, top_k: int = 5) -> List[Document]:
        """
        查询Elasticsearch
        
        Args:
            query: 查询字符串
            top_k: 返回的最相似文档数量
            
        Returns:
            List[Document]: 相关文档列表
        """
        if self.elastic_store is None:
            self.initialize_elastic()
            
            # 如果初始化后仍为None，则返回空列表
            if self.elastic_store is None:
                return []
        
        try:
            return self.elastic_store.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"Error querying Elasticsearch: {e}")
            return []
    
    def query_faiss(self, query: str, top_k: int = 5) -> List[Document]:
        """
        查询FAISS
        
        Args:
            query: 查询字符串
            top_k: 返回的最相似文档数量
            
        Returns:
            List[Document]: 相关文档列表
        """
        if self.faiss_store is None:
            self.initialize_faiss()
            
            # 如果初始化后仍为None，则返回空列表
            if self.faiss_store is None:
                return []
        
        try:
            return self.faiss_store.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"Error querying FAISS: {e}")
            return []
    
    def initialize_compressor(self) -> None:
        """初始化文档压缩器"""
        if self.compressor is None:
            self.compressor = LLMChainExtractor.from_llm(
                llm=self.llm,
            )
        
    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        使用文档压缩器对文档进行压缩
        
        Args:
            documents: 文档列表
            query: 查询字符串
            
        Returns:
            List[Document]: 压缩后的文档列表
        """
        self.initialize_compressor()
        
        try:
            return self.compressor.compress_documents(documents, query)
        except Exception as e:
            print(f"Error compressing documents: {e}")
            return documents
    
    def embedding_compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embedding_model,
            similarity_threshold=USE_EMBEDDING_FOR_COMPRESSION_THRESHOLD,
            )
        return embeddings_filter.compress_documents(documents, query)
    
    def initialize_reranker(self) -> None:
        """初始化重排序器"""
        if self.reranker is None:
            self.reranker = LLMListwiseRerank.from_llm(
                            llm=ChatOpenAI(model=OPEN_AI_MODEL, api_key=OPEN_AI_API_KEY,), 
                            top_n=MAX_DOCUMENTS_AFTER_RERANK
                            )
    def reorder_documents(self, documents: List[Document]) -> List[Document]:
        """
        使用LongContextReorder对文档进行重排序
        
        Args:
            documents: 文档列表
            
        Returns:
            List[Document]: 重排序后的文档列表
        """
        reorderer = LongContextReorder()
        return reorderer.transform_documents(documents)
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        使用LLM对文档进行压缩
        
        Args:
            query: 查询字符串
            documents: 文档列表
            
        Returns:
            List[Document]: 重排序后的文档列表
        """
        self.initialize_reranker()
        self.reranker.compress_documents(documents, query)
        exit()
        try:
            return self.reranker.compress_documents(documents, query)
        except Exception as e:
            print(f"Error reranking documents: {e}")
            return documents
    
    def query(self, query: str, top_k: int = 5) -> List[Document]:
        """
        综合查询Elasticsearch和FAISS，并进行重排序和压缩
        
        Args:
            query: 查询字符串
            top_k: 每个数据源返回的最相似文档数量
            
        Returns:
            List[Document]: 合并、重排序和压缩后的文档列表
        """
        # 查询Elasticsearch
        elastic_docs = self.query_elastic(query, top_k=top_k)
        
        # 查询FAISS
        faiss_docs = self.query_faiss(query, top_k=top_k)
        
        # print("=== Elastic ===")
        # print(len(elastic_docs))
        # print("=== FAISS ===")
        # print(len(faiss_docs))

        # 合并文档
        all_docs = elastic_docs + faiss_docs
        # print("=== All ===")
        # print(len(all_docs))
        # for doc in all_docs:
        #     print(doc.page_content)
        #     print("========================")
        
        if not all_docs:
            return []
        
        # 使用ContextualCompressionRetriever对文档进行压缩
        if not USE_LLM_FOR_COMPRESSION:
            compressed_docs = self.embedding_compress_documents(all_docs, query)
        else:    
            compressed_docs = self.compress_documents(all_docs, query)
        
        
        # print("=== Compressed ===")
        # print(len(compressed_docs))
        # for doc in compressed_docs:
        #     print(doc.page_content)
        #     print("========================")
        
        
        if not USE_LLM_FOR_RERANK:
        # 使用LongContextReorder进行初步重排
            reordered_docs = self.reorder_documents(all_docs)
            # print("=== Reordered ===")
            # print(len(reordered_docs))
            # for doc in reordered_docs:
            #     print(doc.page_content)
            #     print("========================")
        # 使用LLM对文档进行重排序和压缩
        else:
            reordered_docs = self.rerank_documents(query, all_docs)
            # print("=== Reranked ===")
            # print(len(reranked_docs))
            # for doc in reranked_docs:
            #     print(doc.page_content)
            #     print("========================")
        
        return reordered_docs
    
    def extract_info(self, query: str, docs: List[Document]) -> str:
        """
        从文档中提取与查询相关的信息
        
        Args:
            query: 查询字符串
            docs: 文档列表
            
        Returns:
            str: 提取的信息
        """
        if not docs:
            return "未找到相关信息。"
        
        # 合并文档内容
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 使用模板提示LLM提取相关信息
        prompt = f"""根据以下文档内容，提取与查询"{query}"相关的关键信息。
        
文档内容:
{combined_text}

请提取与查询相关的信息，并组织成连贯的回答。如果没有相关信息，请直接说明未找到相关信息。"""
        
        try:
            result = self.llm.invoke(prompt).content
            return result
        except Exception as e:
            print(f"Error extracting information: {e}")
            return "无法提取相关信息，可能是因为信息不足或处理过程出错。"
        
    def _parse_metadata(self, content: str) -> dict:
        metadata = {}
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                key = key.strip()
                value = value.strip()
                # 结构化处理不同字段
                if key == "数据集名称":
                    metadata["dataset_name"] = value
                elif key == "结构":
                    metadata["data_structure"] = value
                elif key == "类别":
                    metadata["data_type"] = value
                elif key == "学习方式":
                    metadata["learning_type"] = value
                elif key == "任务":
                    metadata["task_type"] = value
                elif key == "规模":
                    # 提取样本数和特征数
                    metadata["sample_size"] = value
                elif key == "领域":
                    # metadata["domains"] = [v.strip() for v in re.split(r'[、，]', value) if v.strip()]
                    metadata["domains"] = value
                elif key == "适用目的":
                    # metadata["use_cases"] = [v.strip() for v in re.split(r'[，、]', value) if v.strip()]
                    metadata["use_cases"] = value
                elif key == "适用模型":
                    # 处理多种分隔符
                    # models = [re.sub(r'\s+', ' ', m.strip()) for m in re.split(r'[，,]\s*', value) if m.strip()]
                    metadata["applicable_models"] = value
                elif key == "AI描述":
                    metadata["ai_description"] = value
                elif key == "人工描述":
                    metadata["human_description"] = value
                    # 提取特征详细信息
                    # features = re.findall(r'（([^)]+)）', value)
                    # if features:
                    #     metadata["feature_details"] = [f + " (厘米)" for f in features]

        return metadata

if __name__ == '__main__':
    # 测试数据库查询工具
    query_tool = DBQueryTool()
    query_tool.initialize_elastic()
    query_tool.initialize_faiss()
    result = query_tool.query("数据集", top_k=2)
    # print(result)