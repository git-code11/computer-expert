import typing as tp
from dataclasses import dataclass
import os
import pathlib
import pandas as pd
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


@dataclass
class CropDetail:
    idx: int
    crop: str
    disease: str
    symptoms: str
    causes: str
    solutions: str
    sources: list[str]


class CropDataLoader(BaseLoader):

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

    def _load_data(self, value: tuple[int, pd.Series]) -> CropDetail:
        """
        Converts the raw series to structured data
        Strip the characters included in `STRIP_CHARACTERS` from the `Sources` field
        """
        idx, data = value
        STRIP_CHARACTERS = '• '
        print(data['Source'])
        sources = data['Source'].split('\n')
        sources = list(map(
            lambda x: str.strip(x, STRIP_CHARACTERS),
            sources
        ))
        return CropDetail(
            idx=idx,
            crop=data['Crop'],
            disease=data['Disease'],
            symptoms=data['Visible Symptoms'],
            causes=data['Causes'],
            solutions=data['Solutions'],
            sources=sources
        )

    def _tranform_data(self, data: CropDetail) -> Document:
        """
        Transform the dataset to a valid Document for use with langchain
        """
        page_content = f"Crop Name: {data.crop}\n" \
            f"Diseases: {data.disease}\n" \
            f"Visible Symptoms: {data.symptoms}\n" \
            f"Causes: {data.causes}\n" \
            f"Solutions: {data.solutions}\n" \
            f"Sources/Links: {data.sources}"

        metadata = dict(
            crop=data.crop,
            data=data
        )

        return Document(
            page_content=page_content.strip(),
            metadata=metadata
        )


BASE_DIR = pathlib.Path.cwd()
filepath = BASE_DIR / "crop.xlsx"

# df = pd.read_excel(filepath)
# column_headers = df.columns.to_list()
# print(f"Headers: {', '.join(column_headers)}")

loader = CropDataLoader(filepath)
dataset = loader.load()
print(dataset)
