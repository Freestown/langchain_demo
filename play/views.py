from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
# 初始化环境
import os

@api_view(['POST'])
def play(request):
  os.environ['OPENAI_API_KEY'] = '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'
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
  human_input = request.POST.get('input')
  ai_output = llm([HumanMessage(content=human_input)])
  print(ai_output.content)
  return Response(ai_output.content)


@api_view(['POST'])
def knowledge(request):
  os.environ['OPENAI_API_KEY'] = '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'

  # 加载文档
  loader = TextLoader('play/resources/gpt.txt', encoding='utf8')
  documents = loader.load()

  # 文本分割
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  split_docs = text_splitter.split_documents(documents)

  # 嵌入模型
  embeddings = OpenAIEmbeddings()

  # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
  docsearch = Chroma.from_documents(split_docs, embeddings)

  # 通过向量存储初始化检索器
  retriever = docsearch.as_retriever(search_kwargs={"k": 1})

  qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
  query = request.POST.get('input')
  print(qa(query)["result"])
  return Response(qa(query)["result"])

