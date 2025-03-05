import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
from openai import OpenAI
from baidusearch.baidusearch import search
import requests
from bs4 import BeautifulSoup
from typing import List
from datetime import datetime
import copy
import uvicorn
import re
from multiprocessing import Pool, Manager
import json

app = FastAPI()

n_open_url = 8  # 打开的搜索结果链接数
n_search_words = 4  # 生成的搜索关键词数
n_words_single_page = 1000  # 单个网页提取的文本长度
showmodelname = "OnlineSearchBot-deepseek"


def get_r1_prompt():
    cur_date = datetime.today().strftime('%Y-%m-%d')
    r1prompt = f"""
    在我给你的搜索结果中，每个结果都是[webpage X begin]...[webpage X end]格式的，X代表每篇文章的数字索引。请在适当的情况下在句子末尾引用上下文。请按照引用编号[citation:X]的格式在答案中对应部分引用上下文。如果一句话源自多个上下文，请列出所有相关的引用编号，例如[citation:3][citation:5]，切记不要将引用集中在最后返回引用编号，而是在答案对应部分列出。
    在回答时，请注意以下几点：
    - 今天是{cur_date}。
    - 并非搜索结果的所有内容都与用户的问题密切相关，你需要结合问题，对搜索结果进行甄别、筛选。
    - 对于列举类的问题（如列举所有航班信息），尽量将答案控制在10个要点以内，并告诉用户可以查看搜索来源、获得完整信息。优先提供信息完整、最相关的列举项；如非必要，不要主动告诉用户搜索结果未提供的内容。
    - 对于创作类的问题（如写论文），请务必在正文的段落中引用对应的参考编号，例如[citation:3][citation:5]，不能只在文章末尾引用。你需要解读并概括用户的题目要求，选择合适的格式，充分利用搜索结果并抽取重要信息，生成符合用户要求、极具思想深度、富有创造力与专业性的答案。你的创作篇幅需要尽可能延长，对于每一个要点的论述要推测用户的意图，给出尽可能多角度的回答要点，且务必信息量大、论述详尽。
    - 如果回答很长，请尽量结构化、分段落总结。如果需要分点作答，尽量控制在5个点以内，并合并相关的内容。
    - 对于客观类的问答，如果问题的答案非常简短，可以适当补充一到两句相关信息，以丰富内容。
    - 你需要根据用户要求和回答内容选择合适、美观的回答格式，确保可读性强。
    - 你的回答应该综合多个相关网页来回答，不能重复引用一个网页。
    - 除非用户要求，否则你回答的语言需要和用户提问的语言保持一致。
    - 如果没有搜到和回答用户需求相关的资料，直接指出。严格依据搜索到的内容回答，如搜到的内容可疑可以指出，但不要依据自己的见解回答。
    """
    return r1prompt


# def replace_citations(text, myurl):   #原位替换
#     # 定义一个正则表达式模式来匹配[citation:{数字}]
#     pattern = r'\[citation:(\d+)\]'
#     # 使用re.sub进行替换，lambda函数用于根据匹配到的数字生成替换字符串
#     replaced_text = re.sub(pattern, lambda match: f'[{myurl[int(match.group(1))]}]', text)
#     return replaced_text

def replace_citations(text, myurl):
    # 定义一个正则表达式模式来匹配[citation:{数字}]
    pattern = r'\[citation:(\d+)\]'
    
    # 找到所有引用的位置并去重
    citations = list(set(re.findall(pattern, text)))
    
    # 创建一个字典，将原始引用编号映射到新的序号
    citation_map = {citation: idx + 1 for idx, citation in enumerate(citations)}
    
    # 替换原文中的引用为新的序号
    replaced_text = re.sub(pattern, lambda match: f'[{citation_map[match.group(1)]}]', text)
    
    # 在文本末尾添加引用链接
    references = "\n".join([f"[{citation_map[citation]}] {myurl[int(citation)]}" for citation in citations])
    final_text = f"{replaced_text}\n参考资料:\n{references}"
    
    return final_text



client = OpenAI(
    api_key="123",
    base_url="http://127.0.0.1:8898"
)

class OpenAIChatMessage(BaseModel):
    role: str  # user, assistant, system
    content: Any  # 可以是字符串，也可以是包含多模态数据的列表

class OpenAIChatCompletionRequest(BaseModel):
    # model: str
    messages: List[OpenAIChatMessage]
    # max_tokens: Optional[int] = 4096
    # temperature: Optional[float] = 0.7
    # top_p: Optional[float] = 1.0
    # n: Optional[int] = 1
    # stop: Optional[List[str]] = None
    # stream: Optional[bool] = False

def generate_search_keywords(user_input) -> List[str]:
    cur_date = datetime.today().strftime('%Y-%m-%d')
    search_prompt = f"""
请根据用户的问题生成一组适合在线搜索的关键词。你的目标是提取出最能代表用户查询意图的核心词汇，同时考虑相关的同义词和变体，以保证搜索结果的相关性和全面性。请遵循以下指南来创建搜索关键词：

- 分析用户提问中的关键概念和主题。
- 识别并包含可能影响搜索结果的专业术语或特定领域词汇。
- 考虑添加地理位置、时间范围或其他限定条件，如果它们对回答问题很重要的话。
- 如果适用，包含常见的缩写词或行业简称。
- 避免使用过于宽泛或无关紧要的词汇，以免影响搜索结果的质量。
- 禁止使用知识中的过时内容作为解决有时效性问题的参考。注意当前日期为{cur_date}
- 对于较简单的问题，也不必设计的太复杂
- 由于你得到的是聊天记录或群聊信息，在思考用户需求时，较近的消息或和最后一条消息相关的内容可能拥有更高的权重，最后要求搜索的消息当然最为重要

例如，如果用户询问“2025年最新的AI技术趋势”，你应该生成类似于“2025 AI technology trends\n最新人工智能发展趋势\nAI行业未来预测”的关键词组合。
根据用户本次的搜索目的撰写不超过{n_search_words}个适合浏览器搜索的关键词，每行一个，不要包含其他内容。
"""


    user_input_deep_copy = copy.deepcopy(user_input)

    # 构造提示词，要求生成搜索关键词
    # prompt = f"{user_input}\n\n根据最近的聊天整理用户本次搜索目的并简洁而完整的表述，不要包含其他内容"
    user_input_deep_copy.append({"role": "user", "content": search_prompt})
    
    # 假设 client 已经定义，并且可以调用 chat.completions.create 方法
    completion = client.chat.completions.create(
        model="123",  # 这里应替换为实际使用的模型ID
        messages=user_input_deep_copy
    )
    
    # 提取生成的内容
    raw_content = completion.choices[0].message.content.strip()
    
    # 移除 <think> 标签及其内容
    if "<think>" in raw_content and "</think>" in raw_content:
        start_index = raw_content.find("<think>")
        end_index = raw_content.find("</think>") + len("</think>")
        clean_content = raw_content[:start_index] + raw_content[end_index:]
    else:
        clean_content = raw_content
    
    # 去掉开头结尾的换行，并按行分割
    keywords = [line.strip() for line in clean_content.split('\n') if line.strip()]
    
    return keywords

def fetch_webpage_text(url: str) -> str:
    # 抓取网页内容
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 提取网页文本
    text = soup.get_text()
    return text

def construct_knowledge_prompt(keywords: List[str], myurl) -> str:
    # 构造知识库提示词
    cite_count = 0
    knowledge_prompt = "以下是搜索结果：\n"
    for keyword in keywords:
        knowledge_prompt += f"搜索{keyword}：【"
        # 执行搜索
        results = search(keyword, num_results=n_open_url)
        for result in results:
            # 抓取网页内容并提取文本
            try:
                text = fetch_webpage_text(result['url'])
                myurl.append(result['url'])
                knowledge_prompt += f"[webpage {cite_count} begin]标题：{result['title']}\n网址：{result['url']}\n内容: {text[:n_words_single_page]}...[webpage {cite_count} end]\n\n"  # 只取前500个字符
                cite_count = cite_count + 1
            except Exception as e:
                print(f"\033[91mError fetching {result['url']}: {e}\033[0m")
        knowledge_prompt += f"】\n\n"
    return knowledge_prompt

class TextRequest(BaseModel):
    text: str
@app.post("/simple-chat/")
async def simple_chat_endpoint(request: TextRequest):
    # try:
        # 获取用户输入的文本

        history = []
        lines = request.text.strip().split('\n')
        for line in lines:
            # 跳过空行或仅包含空白字符的行
            if not line.strip():
                continue
            data = json.loads(line)
            history.append(data)

        print(history)
        user_input = history

        # 生成搜索关键词
        keywords = generate_search_keywords(user_input)

        myurl = []

        knowledge_prompt = construct_knowledge_prompt(keywords, myurl)

        r1prompt = get_r1_prompt()

        final_prompt = f"{knowledge_prompt} {r1prompt}"

        # # 将知识库提示词添加到用户输入中
        history.append({"role": "user", "content": final_prompt})
        
        # 调用 OpenAI 模型生成最终输出
        completion = client.chat.completions.create(
            model="123",
            messages=history
        )

        try:
            updated_text = replace_citations(completion.choices[0].message.content, myurl)
        except Exception as e:
            updated_text = completion.choices[0].message.content
        
        # 返回 OpenAI 兼容的响应

        debuginfo0 = "\n".join([f"{i+1}. {keyword}" for i, keyword in enumerate(keywords)])

        # print(f"生成搜索词：\n{debuginfo0}\n最终提示词：{final_prompt}\n")
        print(updated_text)

        # 返回结果
        # return {
        #     "search_keywords": keywords,
        #     "model_output": completion.choices[0].message.content
        # }
        return {
            "result": f"生成搜索词：\n{debuginfo0}\n\n输出：{updated_text}"
        }
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

# #做一个通用的接口以查看发来的包，允许接收任意请求而非OpenAIChatCompletionRequest
# @app.post("/v1/chat/completions")
# async def openai_chat_completion(request: Request):
#     print(request)
#     return {
#         "id": "chatcmpl-12345",
#         "object": "chat.completion",
#         "created": int(time.time()),
#         # "model": request.model,
#         "choices": [
#             {
#                 "index": 0,
#                 "message": {
#                     "role": "assistant",
#                     "content": f"测试消息"
#                 },
#                 "finish_reason": "stop",
#             }
#         ],
#         "usage": {
#             "prompt_tokens": 0,  # 未实现 token 计数
#             "completion_tokens": 0,
#             "total_tokens": 0,
#         },
#     }    


@app.post("/v1/chat/completions")
async def openai_chat_completion(request: OpenAIChatCompletionRequest):
    try:       
        # 生成搜索关键词
        keywords = generate_search_keywords(request.messages)
        
        # 构造知识库提示词
        myurl = []

        knowledge_prompt = construct_knowledge_prompt(keywords, myurl)

        r1prompt = get_r1_prompt()

        final_prompt = f"{knowledge_prompt} {r1prompt}"
        
        # # 将知识库提示词添加到用户输入中
        request.messages.append({"role": "user", "content": final_prompt})
        
        # 调用 OpenAI 模型生成最终输出
        completion = client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            n=request.n,
            stop=request.stop
            # stream=request.stream
        )
        
        # 返回 OpenAI 兼容的响应

        debuginfo0 = "\n".join([f"{i+1}. {keyword}" for i, keyword in enumerate(keywords)])

        try:
            updated_text = replace_citations(completion.choices[0].message.content, myurl)
        except Exception as e:
            updated_text = completion.choices[0].message.content
        

        # print(f"生成搜索词：\n{debuginfo0}\n最终提示词：{final_prompt}\n")
        print(updated_text)

        return {
            "id": "chatcmpl-12345",
            "object": "chat.completion",
            "created": int(time.time()),
            # "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        # "content": completion.choices[0].message.content,
                        # "content": f"用户搜索目的：{search_purpose}\n生成搜索词：\n{debuginfo0}\n最终提示词：{final_prompt}"
                        "content": f"生成搜索词：\n{debuginfo0}\n\n输出：{updated_text}"
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # 未实现 token 计数
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def openai_models():
    return {
        "object": "list",
        "data": [
            {
                "id": showmodelname,
                "object": "model",
                "created": 0,  # 未实现
                "owned_by": "your-organization",
            }
        ],
    }

# 运行应用
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8891)