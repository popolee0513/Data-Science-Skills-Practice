import os
import random
import re
import sys
import numpy as np
DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    corpus = crawl(sys.argv[1])
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    ans=dict()
    outgoing_links=len(corpus[page])
    for i in corpus.keys():
        ans[i]=(1-damping_factor)/len(corpus)
    if outgoing_links!=0:
        for i in corpus[page]:
            ans[i]+=damping_factor/outgoing_links
    return ans

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ans={}
    if n==1:
        temp=random.choice(list(corpus.keys()))
        for i in corpus.keys():
            ans[i]=0
        ans[temp]=1
        return ans
    else:
        final=[]
        temp=np.random.choice(list(corpus.keys()))
        page=temp
        ans=transition_model(corpus, page, damping_factor).values()
        final.append(list(ans))
        new=corpus
        for i in range(n-1):
            temp=np.random.choice(list(corpus.keys()),p=list(ans))
            page=temp
            ans=transition_model(corpus, page, damping_factor).values()
            final.append(list(ans))
        value=np.sum(np.array(final),axis=0)/n
        for i in range(len(corpus)):
             new[list(corpus.keys())[i]]=value[i]
        return new
            
    


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N=len(corpus)
    d=damping_factor
    final=dict()
    for i in list(corpus.keys()):
        final[i]=1/N
    track=final.copy()
    for i in list(corpus.keys()):
        final[i]=0
    for i in list(corpus.keys()):
        for j in list(corpus.keys()):
             if i in list(corpus[j]):
                final[i]+=d*(track[j]/len(corpus[j]))
        final[i]+=(1-d)/N

    while True in list(np.absolute(np.array(list(track.values()))-np.array(list(final.values())))>=0.001):
        track=final.copy()
        for i in list(corpus.keys()):
            final[i]=0
        for i in list(corpus.keys()):
            for j in list(corpus.keys()):
                if i in list(corpus[j]):
                    final[i]+=d*(track[j]/len(corpus[j]))
            final[i]+=(1-d)/N     
    return final

if __name__ == "__main__":
    main()
