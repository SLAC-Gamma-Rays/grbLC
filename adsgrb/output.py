import os


def savePDF(articlelist, output="./"):
    if isinstance(articlelist, str):
        return
    os.makedirs(os.path.dirname(output), exist_ok=True)
    for article, title, year in articlelist:
        with open(output + f"{year}_{title[0][:30].replace(' ', '_').replace('/','-')}.pdf", "wb") as f:
            f.write(article)
