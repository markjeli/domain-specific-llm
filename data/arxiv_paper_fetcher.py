import csv
import logging
import pathlib
import re

import arxiv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

PAGE_SIZE = 10
MAX_RESULTS = 10
SEARCH_QUERY = "medical OR healthcare OR medicine OR disease OR pharmaceutical OR surgery OR x-ray OR cancer OR tomography OR morphology"


def remove_latex_tags(text: str):
    # Remove LaTeX commands but keep the content inside curly braces
    text = re.sub(r"\\[a-zA-Z]+\{", "{", text)

    # Remove itemize
    text = re.sub(r"\{itemize\}", "", text)

    # Remove \item tags
    text = re.sub(r"\\item ", "", text)

    # Remove curly braces but keep the content inside
    text = re.sub(r"\{([^}]*)\}", r"\1", text)

    # Remove LaTeX math environments
    text = re.sub(r"\$.*?\$", "", text)

    # Replace newline characters with a space
    text = re.sub(r"\n+", " ", text)

    return text


if __name__ == "__main__":
    client = arxiv.Client(page_size=PAGE_SIZE, delay_seconds=3.5, num_retries=3)
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # Newest first
    )

    csv_path = pathlib.Path("arxiv_abstracts.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Abstract", "Link"])

        for result_count, result in enumerate(client.results(search)):
            parsed_abstract = remove_latex_tags(result.summary)
            writer.writerow([parsed_abstract, result.pdf_url])

            if (result_count + 1) % PAGE_SIZE == 0:
                logging.info(f"Downloaded {result_count + 1} articles.")
