from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]

files = [Document(page_content = i) for i in texts]
embedding_model = HuggingFaceEmbeddings(
    model_name='E:/BJ/Recipe-RAG/code/C1/hf_cache/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = FAISS.from_documents(files, embedding_model)
faiss_save_path = 'E:/BJ/Recipe-RAG/code/C2/faiss_save'
vectorstore.save_local(faiss_save_path)

print('已成功使用FAISS向量数据库将向量化后的数据存到本地')

load_vectorstore = FAISS.load_local(
    faiss_save_path,
    embedding_model,
    allow_dangerous_deserialization=True
)

query = 'FAISS是做什么的？'

results = load_vectorstore.similarity_search(query, k=1)

print('查询结果问题：', query)
print('相似度最高的文档：', results[0].page_content)