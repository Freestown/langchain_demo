from rest_framework.decorators import api_view
from rest_framework.response import Response
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain


# 常规问答
@api_view(['POST'])
def play(request):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    human_input = request.POST.get('input')
    ai_output = llm([HumanMessage(content=human_input)])
    print(ai_output.content)
    return Response(ai_output.content)


# 基于知识库问答
@api_view(['POST'])
def knowledge(request):
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

    # 通过创建检索器问题对象chain来
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    query = request.POST.get('input')
    ai_output = qa(query)["result"]
    print(ai_output)
    return Response(ai_output)


# 基于数据库问答
@api_view(['POST'])
def database(request):
    db = SQLDatabase.from_uri("mysql+pymysql://root:12345678@127.0.0.1/zyl")
    llm = OpenAI(temperature=0)
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    query = request.POST.get('input')
    ai_output = db_chain.run(query)
    print(ai_output)
    return Response(ai_output)
