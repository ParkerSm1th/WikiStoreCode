"""This is the logic for ingesting Notion data into LangChain."""
import pickle
import sys
from pathlib import Path

import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

spaceId = sys.argv[1];

# Here we load in the data in the format that Notion exports it in.
ps = list(Path(f'WikiStore/{spaceId}/').glob("**/*.md"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, f'faiss_store_{spaceId}.index')
store.index = None
with open(f'faiss_store_{spaceId}.pkl', "wb") as f:
    pickle.dump(store, f)
