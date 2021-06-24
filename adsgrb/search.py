import requests
from .config import read_apikey
from .ECHO import SynchronizedEcho
import concurrent.futures, warnings
from ads import SearchQuery


def getArticles(papers, threading=True, debug=False):
    """
    User function to create a single string containing seperated text bodies from a
    list of `ads.search.Article`'s.

    :param papers:
        A list of ADS articles to download.
    :type papers:
        :class:`list` of `ads.search.Article`
    :param threading:
        Boolean to specify the use of concurrency.
    :type threading:
        :class:`bool`
    :returns:
        String containing each GCN separated by a line.
    """
    if len(papers) == 0:
        return r"No articles found! ¯\(°_o)/¯"

    articlelist = []
    if threading:
        threads = min(30, len(papers))
        _wrapped_getArticle = lambda article: getArticle(articlelist, article, debug=debug)

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            executor.map(_wrapped_getArticle, papers)
            executor.shutdown()
    else:
        articlelist = [getArticle(articlelist, paper, debug=debug) for paper in papers]

    if "gcn" in papers[0].bibcode.lower():
        result = "\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n\n".join(articlelist)
    else:
        result = articlelist

    return result


def getGRBComboQuery(GRB):
    """
    Get the several versions of a GRB name that could come up in ADS searches.
    E.g., 010222A, 10222A, GRB010222A, GRB1022A

    :param GRB:
        The GRB to get name combinations of.
    :type GRB:
        :class:`str`
    :returns:
        String of GRB name combinations separated by "OR" for search in ADS.
    """

    return " OR ".join([f"{GRB}", f"GRB{GRB}"])


def additionalKeywords(keywords):
    """
    Convert keyword(s) to a string to use in an ADS query.

    :param keywords:
        Keywords to specifically search for in addition to the GRB.
    :type keywords:
        :class:`list`,`tuple`,`str`
    :returns:
        String of keyword(s) separated by an "AND" for use in an ADS query.
    """

    if not isinstance(keywords, (type(None), list, tuple)):
        keywords = (keywords,)

    return " AND " + " AND ".join(keywords) if keywords else ""


def gcnSearch(GRB, keywords=None, printlength=True, debug=False):
    """
    User function to find GCNs containing the inputted GRB and optional
    keywords

    :param GRB:
        GRB name; e.g., '010222' or '200205A'
    :type GRB:
        :class:`str`
    :param keywords:
        Keywords to specifically search for in addition to the GRB.
    :type keywords:
        :class:`list`,`tuple`,`str`
    :param printlength:
        Determines whether the user would like the number of articles found to be printed.
    :type printlength:
        :class:`bool`
    :returns:
        A list of `ads.search.Article`'s containing GCNs pertaining to GRB and optional
        keywords.
    """

    if keywords is not None:
        warnings.warn("Keywords aren't working correctly right now.", stacklevel=2)
    assert isinstance(GRB, str), "GRB is not of type string."
    query = f"bibstem:GCN {getGRBComboQuery(GRB)}"
    keywords = additionalKeywords(keywords)
    finds = list(SearchQuery(q=f"{query + keywords}", fl=["bibcode", "identifier"]))
    if debug:
        print(f"[ADSGRB] Query: {query + keywords}")
    if printlength:
        print(f"[{GRB}] {len(finds)} entries found.")
    return finds


def litSearch(GRB, keywords=None, printlength=True, debug=False):
    """
    User function to find literature containing the inputted GRB and optional
    keywords

    :param GRB:
        GRB name; e.g., '010222' or '200205A'
    :type GRB:
        :class:`str`
    :param keywords:
        Keywords to specifically search for in addition to the GRB.
    :type keywords:
        :class:`list`,`tuple`,`str`
    :param printlength:
        Determines whether the user would like the number of articles found to be printed.
    :type printlength:
        :class:`bool`
    :returns:
        A list of `ads.search.Article`'s containing GCNs pertaining to GRB and optional
        keywords.
    """

    if keywords is not None:
        warnings.warn("Keywords aren't working correctly right now.", stacklevel=2)
    assert isinstance(GRB, str), "GRB is not of type string."
    query = getGRBComboQuery(GRB)
    keywords = additionalKeywords(keywords)
    querysnippet = f"(grb AND {query + keywords})"
    fullquery = f"title:{querysnippet} OR abstract:{querysnippet} OR keyword:{querysnippet} doctype:article"
    finds = list(SearchQuery(q=fullquery, fl=["bibcode", "identifier", "title", "author", "year"], rows=100))
    if debug:
        print(f"[ADSGRB] Query: {fullquery}")
    if printlength:
        print(f"[{GRB}] {len(finds)} entries found.")
    return finds


def getArticle(articlelist, article, debug=False):
    """
    Download an article from arXiv or other sources.

    :param articlelist:
        The string list to append article texts to.
    :type articlelist:
        :class:`list`
    :param article:
        The ADS article to retrieve.
    :type article:
        :class:`ads.search.Article`
    :param debug:
        Optional debugging
    :type debug:
        :class:
    :returns:
        Nothing. Side effect of appending text of article body to articlelist.

    Modified from https://github.com/andycasey/ads/blob/master/examples/monthly-institute-publications/stromlo.py#22
    """

    if debug:
        ECHO("[ADSGRB] Retrieving {0}".format(article.bibcode))
    isGCN = "GCN" in article.bibcode
    header = {"Authorization": f"Bearer {read_apikey()}"}
    # Ask ADS to redirect us to the journal article.
    if isGCN:
        params = {"bibcode": article.bibcode, "link_type": "EJOURNAL"}
    else:
        params = {"bibcode": article.bibcode, "link_type": "ESOURCE"}

    url = requests.get("http://adsabs.harvard.edu/cgi-bin/nph-data_query", params=params).url

    if isGCN:
        q = requests.get(url, allow_redirects=True)
    else:
        q = requests.get(
            f"https://api.adsabs.harvard.edu/v1/resolver/{article.bibcode}/esource",
            headers=header,
            allow_redirects=False,
        )
        if not q.ok:
            ECHO("Error retrieving {0}: {1} for {2}".format(article, q.status_code, url))
            if debug:
                q.raise_for_status()
            else:
                return

        deserialized = q.json()
        try:
            records = deserialized["links"]["records"]
            for record in records:
                if "PDF" in record["link_type"]:
                    ECHO(record["url"].replace("arxiv.org", "export.arxiv.org"))
                    # switch any arxiv url to export.arxiv so we don't get locked out
                    url = record["url"].replace("arxiv,org", "export.arxiv.org")
                    q = requests.get(url, stream=True)
                    break
        except:
            # switch any arxiv url to export.arxiv so we don't get locked out
            url = deserialized["link"].replace("arxiv.org", "export.arxiv.org")
            q = requests.get(url, stream=True)

    if not q.ok:
        ECHO("Error retrieving {0}: {1} for {2}".format(article, q.status_code, url))
        if debug:
            q.raise_for_status()
        else:
            return
    # Check if the journal has given back forbidden HTML.
    try:
        if q.content.endswith("</html>"):
            ECHO("Error retrieving {0}: 200 (access denied?) for {1}".format(article, url))
            return
    except:
        if q.text.endswith("</html>"):
            ECHO("Error retrieving {0}: 200 (access denied?) for {1}".format(article, url))
            return

    if isGCN:
        articlelist.append(q.text)
    else:
        articlelist.append([q.content, article.title, article.year])


ECHO = SynchronizedEcho()
