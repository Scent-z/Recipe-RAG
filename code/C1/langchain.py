import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader  # LangChain 对 Unstructured 库的封装
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

os.environ['HUGGINGFACE_HUB_CACHE'] = './hf_cache'

load_dotenv()

markdown_path = 'E:/BJ/Recipe-RAG/data/C1/markdown/easy-rl-chapter1.md'
loader = UnstructuredMarkdownLoader(markdown_path)
files = loader.load()

splitter = RecursiveCharacterTextSplitter()  # 默认参数 chunk_size=4000（块大小）和 chunk_overlap=200（块重叠）
chunks = splitter.split_documents(files)  # 每个chunk是一个Document对象，包括metadata（数据来源）与page_content（当前块的文本内容）

embedding_model = HuggingFaceEmbeddings(
    model_name='E:/BJ/Recipe-RAG/code/C1/hf_cache/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = InMemoryVectorStore(embedding_model)
vectorstore.add_documents(chunks)

llm = ChatOpenAI(
    model='glm-4.7-flash-free',
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv('AIHUBMIX_API_KEY'),
    base_url='https://aihubmix.com/v1'
)
 
prompt = ChatPromptTemplate.from_template(
    """
    请根据下面提供的上下文信息来回答问题。
    请确保你的回答完全基于这些上下文。
    如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

    上下文:{context}

    问题: {question}

    回答:
    """
)

question = "文中举了哪些例子？"
retrieved_files = vectorstore.similarity_search(question, k=3)

# 使用 "\n\n" (双换行符) 而不是 "\n" (单换行符) 来连接不同的检索文档块，主要是为了在传递给大型语言模型（LLM）时，能够更清晰地在语义上区分这些独立的文本片段。
# 双换行符通常代表段落的结束和新段落的开始，这种格式有助于LLM将每个块视为一个独立的上下文来源，从而更好地理解和利用这些信息来生成回答
content = '\n\n'.join(f.page_content for f in retrieved_files)

response = llm.invoke(prompt.format(context=content, question=question))

print(response.content)