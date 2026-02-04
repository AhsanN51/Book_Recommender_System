import pandas as pd
import numpy as np
from dotenv import load_dotenv

import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

books = pd.read_csv("cleaned_classified_withsemanys_books.csv")

books["enlarged_thumbnails"] = books["thumbnail"] + "&fifi=w800"
books["enlarged_thumbnails"] = np.where(
    books["enlarged_thumbnails"].isna(), "images.png", books["enlarged_thumbnails"]
)


raw_documents = TextLoader("tagged_isbn13_description.txt", encoding="utf-8").load()
splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
documents = splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, MistralAIEmbeddings())


def ret_sem_rec(q, c, m, i_t_k=48, f_t_k=16):

    sim_des_srls = db.similarity_search(q, k=i_t_k)
    rec_books_ids = [
        int(sim_des_srb.page_content.strip().split()[0]) for sim_des_srb in sim_des_srls
    ]
    rec_books = books[books["isbn13"].isin(rec_books_ids)].head(i_t_k)

    if c != "All":
        rec_books = books[books["simple_categories"] == c].head(f_t_k)
    else:
        rec_books = books.head(f_t_k)

    if m == "Happy":
        rec_books.sort_values(by="joy", ascending=False, inplace=True)
    elif m == "Anger":
        rec_books.sort_values(by="anger", ascending=False, inplace=True)
    elif m == "Thriller":
        rec_books.sort_values(by="fear", ascending=False, inplace=True)
    elif m == "Sadness":
        rec_books.sort_values(by="sadness", ascending=False, inplace=True)
    elif m == "Surprise":
        rec_books.sort_values(by="suprise", ascending=False, inplace=True)

    return rec_books


def recommenders(query, category, mood):

    recommendations = ret_sem_rec(query, category, mood)
    res = []

    for _, book in recommendations.iterrows():

        split_book_des = book["description"].split()
        trunc_bookdes = " ".join(split_book_des[:30]) + "....."

        authors_split = book["authors"].split(";")
        if len(authors_split) >= 2:
            authorstr = f"{','.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authorstr = book["authors"]

        caption = f"{book['title']} by {authorstr}: \n {trunc_bookdes}"

        res.append((book["enlarged_thumbnails"], caption))

    return res


categories = ["All"] + sorted(books["simple_categories"].unique())
mood = ["All"] + ["Happy", "Anger", "Thriller", "Sadness", "Surprise"]

with gr.Blocks() as interface:

    gr.Markdown("# Book Recommendation System 7k")

    with gr.Row():

        des = gr.Textbox(
            label="Please enter description of the desired books",
            placeholder="eg. a story about destiny....",
        )
        cat = gr.Dropdown(label="Select the category", choices=categories, value="All")
        tone = gr.Dropdown(label="Select the tone", choices=mood, value="All")
        submit = gr.Button("Find Recommendations")

    gr.Markdown("## Book Recommendations")

    res = gr.Gallery(label="Recommended Books", columns=8, rows=2)

    submit.click(fn=recommenders, inputs=[des, cat, tone], outputs=res)

if __name__ == "__main__":

    interface.launch(theme=gr.themes.Glass())
