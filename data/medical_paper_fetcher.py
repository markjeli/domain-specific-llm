import csv
import logging
import pathlib

from paperscraper.xrxiv.xrxiv_api import MedRxivApi, BioRxivApi

from arxiv_paper_fetcher import remove_latex_tags

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

BIORXIV_CATEGORIES = [
    "cancer biology",
    "clinical trials",
    "epidemiology",
    "genetics",
    "genomics",
    "immunology",
    "microbiology",
    "molecular biology",
    "neuroscience",
    "pathology",
    "pharmacology",
    "toxicology",
    "physiology",
]


def fetch_papers(api, fields, counter, categories=None):
    for paper in api.get_papers(fields=fields):
        if (categories and paper["category"].lower() not in categories) or paper[
            "doi"
        ] in doi_numbers:
            continue
        doi_numbers.add(paper["doi"])
        parsed_abstract = remove_latex_tags(paper["abstract"])
        try:
            writer.writerow([parsed_abstract, paper["doi"]])
            counter += 1
        except UnicodeEncodeError:
            logging.error(f"Failed to write paper with DOI: {paper['doi']}")
            continue
        if counter % 1000 == 0:
            logging.info(
                f"Downloaded {counter} articles from {api.__class__.__name__}."
            )
    return counter


if __name__ == "__main__":
    medrxiv_api = MedRxivApi(max_retries=10)
    biorxiv_api = BioRxivApi(max_retries=10)
    csv_path = pathlib.Path("medical_abstracts.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Abstract", "DOI"])
        doi_numbers = set()
        medrxiv_paper_counter = 0
        biorxiv_paper_counter = 0

        # Fetch papers from medRxiv
        medrxiv_paper_counter = fetch_papers(
            medrxiv_api, ["abstract", "doi"], medrxiv_paper_counter
        )
        # Fetch papers from bioRxiv
        biorxiv_paper_counter = fetch_papers(
            biorxiv_api,
            ["abstract", "doi", "category"],
            biorxiv_paper_counter,
            BIORXIV_CATEGORIES,
        )

    logging.info(
        f"Fetching done downloading {medrxiv_paper_counter + biorxiv_paper_counter} articles."
    )
