"""
MCP (Model Context Protocol) 查询工具
自定义工具，用于从MCP API获取外部信息
"""
import json
import requests
from typing import Dict, Any, Optional
from urllib.parse import quote

from config import MCP_API_URL, MCP_API_KEY

class MCPQueryTool:
    """MCP查询工具类"""
    
    def __init__(self):
        """初始化MCP查询工具"""
        self.api_url = MCP_API_URL
        self.api_key = MCP_API_KEY
        
        # 检查API密钥
        if not self.api_key:
            print("Warning: MCP API key is not set.")
    
    def search(self, query: str) -> Dict[str, Any]:
        """
        通过MCP API搜索信息
        
        Args:
            query: 查询字符串
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        if not self.api_key:
            return {
                'success': False,
                'message': 'MCP API密钥未设置',
                'data': None
            }
        
        try:
            # 对查询进行URL编码
            encoded_query = quote(query)
            
            # 构建API请求
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'query': query,
                'max_results': 5
            }
            
            # 发送请求
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'message': '成功获取信息',
                    'data': result
                }
            else:
                return {
                    'success': False,
                    'message': f'API错误: {response.status_code}',
                    'data': None
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'message': f'请求错误: {str(e)}',
                'data': None
            }
        except json.JSONDecodeError:
            return {
                'success': False,
                'message': '无法解析API响应',
                'data': None
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'未知错误: {str(e)}',
                'data': None
            }
    
    def extract_content(self, result: Dict[str, Any]) -> str:
        """
        从API结果中提取内容
        
        Args:
            result: API返回的结果
            
        Returns:
            str: 提取的内容
        """
        if not result.get('success', False):
            return f"无法获取信息: {result.get('message', '未知错误')}"
        
        data = result.get('data', {})
        
        # 根据实际的API响应结构提取信息
        # 这里使用一个通用的方法，实际使用时应根据实际的API响应格式调整
        if isinstance(data, dict):
            content = data.get('content', '')
            sources = data.get('sources', [])
            
            if content:
                if sources:
                    source_text = "\n\n来源:\n" + "\n".join([f"- {s}" for s in sources])
                    return content + source_text
                return content
            
            # 如果没有直接的content字段，尝试从其他字段提取
            results = data.get('results', [])
            if results and isinstance(results, list):
                extracted = []
                for i, item in enumerate(results[:5]):  # 限制最多5个结果
                    if isinstance(item, dict):
                        title = item.get('title', f'结果 {i+1}')
                        snippet = item.get('snippet', '')
                        url = item.get('url', '')
                        
                        item_text = f"### {title}\n{snippet}\n{url}"
                        extracted.append(item_text)
                
                if extracted:
                    return "\n\n".join(extracted)
        
        # 如果无法提取有意义的内容，返回原始数据的字符串表示
        return f"获取到的原始数据: {str(data)}"
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        查询MCP并提取结果
        
        Args:
            query: 查询字符串
            
        Returns:
            Dict[str, Any]: 查询结果
        """
        search_result = self.search(query)
        content = self.extract_content(search_result)
        
        return {
            'success': search_result.get('success', False),
            'message': search_result.get('message', ''),
            'content': content
        }