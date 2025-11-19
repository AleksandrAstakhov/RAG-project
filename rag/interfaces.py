from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List


class AbstractSearcher(ABC):
    @abstractmethod
    def search(self, query: str, max_docs: int = 3) -> List[Document]:
        """Поиск документов по запросу"""
        pass


class AbstractSplitter(ABC):
    @abstractmethod
    def split(self, docs: List[Document]) -> List[Document]:
        """Разделение документов на чанки"""
        pass


class AbstractStorage(ABC):
    @abstractmethod
    def build_index(self, chunks: List[Document]):
        """Создание векторного индекса"""
        pass

    @abstractmethod
    def get_retriever(self, k: int = 4):
        """Получение retriever для поиска по индексу"""
        pass


class AbstractLLM(ABC):
    @abstractmethod
    def __call__(self, prompt: str):
        """Вызов LLM с prompt"""
        pass
