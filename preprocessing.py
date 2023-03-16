from sentence_transformers import SentenceTransformer



def get_pages(path: str) -> dict[str]:
    """
    Read the book text file and return a dictionary mapping
    a page number to page content.
    Page numbers start from 1. A page content is a single string
    composed of multiple sentences.
    """
    with open(path) as f:
        lines = f.readlines()
    lines[0] = lines[0][1:]  # Remove special character at he beginning of the file

    pages = {}

    i = 1
    for line in lines:
        # Split pages
        if line == '________________\n':
            i += 1
        # Ignore empty lines
        elif line == '\n':
            pass
        # Concatenate lines on the same page
        else:
            pages[i] = pages.get(i, '') + line
    
    return pages


def get_sentences_per_page(pages: dict[str]) -> dict[list[str]]:
    """
    
    """

    sentences = {}

    for page_number, page in pages.items():
        page_sentences = [s + '.' if s == '' or s[-1].isalpha() else s for s in page.strip('.\n').split('. ')]
        
        # Split poems into multiple sentences
        for i in range(len(page_sentences)):
            if '\n' in page_sentences[i]:
                page_sentences = (
                    page_sentences[:i] + [s.strip() for s in page_sentences[i].split('\n')] + page_sentences[i+1:]
                )

        # If a dot is on its own, add it to the previous sentence (e.g. "He drops awa....." in the last page)
        i = 0
        while i < len(page_sentences):
            if page_sentences[i] == '.':
                page_sentences[i-1] += '.'
                del page_sentences[i]
            else:
                i += 1
        
        sentences[page_number] = page_sentences
    
    return sentences


def get_embeddings_per_page(sentences: dict[list[str]]):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = {}

    for page_number, sentence_list in sentences.items():
        embeddings[page_number] = [
            model.encode(sentence) for sentence in sentence_list
        ]

    return embeddings