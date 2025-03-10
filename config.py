"""
配置文件，存储各种参数和API密钥
"""
import os
from dotenv import load_dotenv
import pandas as pd 

# 加载环境变量
load_dotenv()

# API配置
if not os.getenv("ZHIPUAI_API_KEY"):
    os.environ["ZHIPUAI_API_KEY"] = '626e93484e6643af9dda2f9deffe803a.rXJLatBAvI7ZwNve'
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

# 模型配置
LLM_MODEL = "glm-4-flash"  # 使用智谱AI的GLM-4模型
EMBEDDING_MODEL = "embedding-3"  # 使用智谱AI的Embedding模型
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
OPEN_AI_MODEL = "gpt-4o"

# FAQ系统配置
FAQ_SIMILARITY_THRESHOLD_K1 = 0.8  # 高于此阈值，直接返回缓存答案给用户
FAQ_SIMILARITY_THRESHOLD_K2 = 0.6  # 低于此阈值，直接发送原始Query给Agent

# 打分模型配置
SCORING_THRESHOLD_K3 = 0.7  # 打分低于此阈值，重新发送给Agent

# 向量数据库配置
FAISS_INDEX_NAME = "faiss_index"
FAISS_CHUNK_SIZE = 1000
FAISS_CHUNK_OVERLAP = 200
FAISS_RECREATE = True

# Elasticsearch配置
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_INDEX = "documents"
ELASTICSEARCH_RECREATE = True

# 缓存配置
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_DIR = os.path.join(DATA_DIR, "csv")
PDF_DIR = os.path.join(DATA_DIR, "pdf")

# vLLM配置
VLLM_MODEL_DIR = os.getenv("VLLM_MODEL_DIR", None)  # 本地模型路径，如果没有则使用API
VLLM_MAX_MODEL_LEN = 8192
VLLM_GPU_MEMORY_UTILIZATION = 0.9

# MCP配置
MCP_API_URL = os.getenv("MCP_API_URL", "https://api.example.com/mcp")
MCP_API_KEY = os.getenv("MCP_API_KEY")

# 重排序与压缩配置
MAX_DOCUMENTS_AFTER_RERANK = 10
USE_LLM_FOR_RERANK = False
USE_LLM_FOR_COMPRESSION = False
USE_EMBEDDING_FOR_COMPRESSION_THRESHOLD = 0.1

# 工具配置
MAX_PAPERS = 10





def load_data_xlsx_to_csv(input_file_path='/Users/utopia/Documents/毕设/claude/rag_project/data/data.xlsx', output_file_path='/Users/utopia/Documents/毕设/claude/rag_project/data/csv/FAQ.csv'):
    """加载FAQ数据并存储在类变量中"""
    df = pd.read_excel(input_file_path)
    documents = df.to_dict('records')  # 按行读取数据
    documents_processed_col = []  # 先进行列处理
    for doc in documents:
        doc = {k: v for k, v in doc.items() if not k.startswith('Unnamed:')}
        documents_processed_col.append(doc)

    documents_processed = []  # 再进行行处理
    for doc in documents_processed_col:
        if pd.isnull(doc['数据集名称']):
            continue
        documents_processed.append(doc)
        
    # 写入csv文件
    df = pd.DataFrame(documents_processed)
    df.to_csv(output_file_path, index=False)
    print(documents_processed[0])
    print(f"数据已保存到: {output_file_path}")

if __name__ == '__main__':
    # load_data_xlsx_to_csv()
    import pandas as pd

    # 读取csv文件
    df = pd.read_csv('/Users/utopia/Documents/毕设/claude/rag_project/data/csv/FAQ.csv', encoding='utf-8')

    # 显示前5行数据
    print(f"数据集形状：{df.shape}")  # 输出示例：(100, 3) 表示100行3列
    print(df.head())
    # print(CACHE_DIR)
    # print(CSV_DIR)
    # print(PDF_DIR)
