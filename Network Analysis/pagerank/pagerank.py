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
    prob =dict()
    
    if len(corpus[page])!=0:
        for key in corpus:
            prob[key]= (1-damping_factor)/len(corpus)
    
        connect = corpus[page]
        for key in connect:
            prob[key]+= damping_factor/len(connect)
        return prob
    else:
        for key in corpus:
            prob[key]= 1/len(corpus)
        return prob


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    count = dict()
    for key in corpus:
        count[key]=0
    page =  random.choice(list(corpus.keys()))
    for i in range(n):
        prob = transition_model(corpus, page, damping_factor)
        page = np.random.choice(list(prob.keys()),p=list(prob.values()))
        count[page]+=1
    for key in count:
        count[key]/= n
    return count



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
   ###A page that has no links at all should be interpreted as having one link for every page in the corpus
    for key in corpus:
        if len(corpus[key])==0:
            value = set(corpus.keys())
            value.remove(key)
            corpus[key]=value
    
    pagerank = dict()
    for key in corpus:
        pagerank[key]=1/len(corpus)
    iteration = True
    while iteration :
        check =[]
        temp_dict  = pagerank.copy()
        for key in pagerank:
            value = (1- damping_factor) / len(corpus)
            temp = list(pagerank.keys())
            temp.remove(key)
            
            for other in temp:
                if key in corpus[other]:
                    value+=damping_factor*(pagerank[other]/len(corpus[other]))
            if np.abs(pagerank[key]-value)<1e-15:
                check.append(False)
            else:
                check.append(True)
            temp_dict[key] = value
        pagerank= temp_dict
        iteration =any(check)
    return pagerank



if __name__ == "__main__":
    main()
