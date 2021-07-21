import os


def savePDF(articlelist, output=os.getcwd(), debug=False):
    if isinstance(articlelist, str):
        return
    os.makedirs(os.path.dirname(output), exist_ok=True)
    for article in articlelist:
        if len(article) == 4:
            txt, title, year, url = article
            with open(output + f"{year}_{title[0][:30].replace(' ','_').replace('/','-')}.pdf", "wb") as f:
                f.write(txt)
            if debug:
                with open(output + f"{year}_{title[0][:30].replace(' ','_').replace('/','-')}.txt", "w") as dbg:
                    dbg.write("URL: " + url)
        else:
            continue
