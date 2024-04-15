import os
from multiprocessing import Pool

import httpx
import torch.backends.mps
import trafilatura
from parsel import Selector
from pymilvus import utility, connections
from tqdm import tqdm

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (messages_to_prompt, completion_to_prompt)
from llama_index.core import set_global_tokenizer, Document, StorageContext, VectorStoreIndex
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.milvus import MilvusVectorStore

def connect_milvus():
    connections.connect(
      alias="default",
      host='localhost',
      port='19530'
    )


def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    return device


def set_embed_model(device):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", device=device)
    Settings.embed_model = embed_model


def check_collection(collection_name):
    connect_milvus()
    return utility.has_collection(collection_name)


def read_web_pages(path):
    urls = []
    with open(path) as f:
        for url in f.readlines():
            urls.append(url.strip())
    documents = load_data(urls=urls)
    return documents


def load_data(urls,
              include_comments=True,
              output_format="txt",
              include_tables=True,
              include_images=False,
              include_formatting=False,
              include_links=False):
    documents = []
    pool_args = [
        (url, include_comments, output_format, include_tables, include_images, include_formatting, include_links) for
        url in urls]

    with Pool() as pool:
        for result in tqdm(pool.imap_unordered(process_url, pool_args), total=len(urls), desc="Fetching documents..."):
            if result is not None:
                documents.append(result)

    return documents


def process_url(args):
    # Unpack all arguments from the single tuple argument
    url, include_comments, output_format, include_tables, include_images, include_formatting, include_links = args
    downloaded = trafilatura.fetch_url(url)
    response = trafilatura.extract(
        downloaded,
        include_comments=include_comments,
        output_format=output_format,
        include_tables=include_tables,
        include_images=include_images,
        include_formatting=include_formatting,
        include_links=include_links,
    )
    if response is None:
        print(f"{url} is empty")
        return None
    else:
        return Document(text=response, id_=url)


def create_db():
    documents = read_web_pages("./data/towson/urls.txt")
    vector_store = MilvusVectorStore(dim=1024, overwrite=True, collection_name="Towson")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    return index


def parse_sitemap(output_dir):
    sitemap_url_path = "https://www.towson.edu/sitemap.xml"
    response = httpx.get(sitemap_url_path)
    selector = Selector(response.text)
    urls = []
    pdfs = []
    for url in selector.xpath('//url'):
        location = url.xpath('loc/text()').get()
        modified = url.xpath('lastmod/text()').get()
        if ".pdf" not in location:
            urls.append(location)
        else:
            pdfs.append(location)
    write_list_to_file(output_dir=output_dir, file_name="urls.txt", data=urls)
    write_list_to_file(output_dir=output_dir, file_name="pdfs.txt", data=pdfs)


def write_list_to_file(output_dir, file_name, data):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, file_name), 'w') as f:
        for item in data:
            f.write(str(item) + '\n')


def initialize_llm():
    model_path = "./models/mistral-7b-instruct-v0.2.Q8_0.gguf"
    llm = LlamaCPP(
        model_path=model_path,
        temperature=0.1,
        max_new_tokens=256,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": -1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt
    )
    return llm


def load_db():
    vector_store = MilvusVectorStore(
        host="localhost",
        port="19530",
        collection_name="Towson"
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


if __name__ == '__main__':
    device = set_device()
    set_embed_model(device)
    if len(os.listdir("./data/towson/")) == 0:
        print("No Towson links found, creating from sitemap...")
        parse_sitemap(output_dir="./data/towson")
    else:
        print("Towson links found")

    check = check_collection("Towson")
    if check is False:
        print("Towson vector database does not exist in milvus, creating new one...")
        index = create_db()
    else:
        print("Towson vector database exists, loading...")
        index = load_db()
    llm = initialize_llm()
    query_engine = index.as_query_engine(llm=llm, streaming=True, similarity_top_k=3)
    streaming_response = query_engine.query("What building is the Towson gym located in?")
    streaming_response.print_response_stream()



