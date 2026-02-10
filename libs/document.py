import typing as tp
from dataclasses import dataclass
import os
import pathlib
import pandas as pd
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


@dataclass
class ItemDetail:
    idx: int
    device: str
    problem: str
    symptoms: str
    diagnosis: str
    solutions: str
    sources: str


class ComputerDataLoader(BaseLoader):

    def __init__(self, filepath: os.PathLike):
        self.filepath = filepath

    def load(self) -> list[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> tp.Iterator[Document]:
        df = pd.read_excel(self.filepath)
        df.dropna(axis=0, inplace=True)
        datas = map(self._load_data, df.iterrows())
        documents = map(self._tranform_data, datas)
        return documents

    def _load_data(self, value: tuple[int, pd.Series]) -> ItemDetail:
        """
        Converts the raw series to structured data
        """
        idx, data = value

        return ItemDetail(
            idx=idx,
            device=data['Device'],
            problem=data['Problem'],
            symptoms=data['Symptoms'],
            solutions=data['Solutions'],
            diagnosis=data['Diagnosis'],
            sources=data['Sources']
        )

    def _tranform_data(self, data: ItemDetail) -> Document:
        """
        Transform the dataset to a valid Document for use with langchain
        """
        page_content = f"Device: {data.device}\n" \
            f"Problem: {data.problem}\n" \
            f"Symptoms: {data.symptoms}\n" \
            f"Solutions: {data.solutions}\n" \
            f"Diagnosis: {data.diagnosis}\n" \
            # f"Sources: {data.sources}"

        metadata = dict(
            device=data.device,
            data=data
        )

        return Document(
            page_content=page_content.strip(),
            metadata=metadata
        )


if __name__ == '__main__':
    BASE_DIR = pathlib.Path.cwd()
    filepath = BASE_DIR / "data.xlsx"

    # df = pd.read_excel(filepath)
    # column_headers = df.columns.to_list()
    # print(f"Headers: {', '.join(column_headers)}")

    loader = ComputerDataLoader(filepath)
    dataset = loader.load()
    print(dataset)
