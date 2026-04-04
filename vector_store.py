from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embeddings
from typing import List
from dotenv import load_dotenv
from os import getenv
from os.path import isfile

load_dotenv()

#문서를 불러와 분할, 임베딩 한 데이터를 VS에 넣고 로컬에 저장, 읽어들이기
#필요기능
#문서 로드
#!큰 문서의 경우 청킹
#문서 임베딩 후 벡터스토어에 담기
#담은 벡터스토어를 로컬에 저장 - 토큰 절약, 시간 등등 이득

#위 과정, 벡터스토어 존재하지 않을 경우 실행
#존재한다면 읽어들이기만 실행

#문서 로딩 후 딕셔너리로 변환
def load_documents()->List[Document]:
    loader = CSVLoader(getenv("FILE_PATH"), encoding="utf-8")

    loaded_docs = loader.load()
    docs = []
    title = ['id','type','title','content','author','created_at']
    for loaded_doc in loaded_docs:
        dic={}
        line = loaded_doc.page_content.split('\n')
        for j in range(6):
            line_content = line[j].replace(title[j]+': ', '')
            dic[title[j]] = line_content
        docs.append(dic)
    

    return [
            Document(
                page_content=f"제목: {doc['title']}\n\n내용: {doc['content']}",
                metadata={
                    'id':doc['id'],
                    'type':doc['type'],
                    'author':doc['author'],
                    'created_at': doc['created_at']
                } 
            )

            for doc in docs
        ]

#임베딩 모델 불러오기
def embedding(docs:List[Document])->FAISS:
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding = embeddings,
    )

    return vector_store

#임베딩된 파일 저장하기
def save_vector_to_local(vectorstore):
    path_str = getenv("SAVE_PATH")
    vectorstore.save_local(path_str)

#벡터스토어 로드 -> 공부 좀 더 하자
def load_vector_to_local() -> FAISS:
    path_str = getenv("SAVE_PATH")
    return FAISS.load_local(
        path_str,
        get_embeddings(),
        allow_dangerous_deserialization=True #?
    )

#벡터스토어 초기화
def init_vectorstore():
    docs = load_documents()
    vectorstore = embedding(docs)
    save_vector_to_local(vectorstore)
    return vectorstore
    
#벡터스토어 반환
def get_vectorstore()->FAISS:
    if isfile(getenv("SAVE_PATH")):
        return load_vector_to_local()
    else:
        return init_vectorstore()