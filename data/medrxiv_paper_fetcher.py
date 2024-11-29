import csv
import logging
import pathlib

from paperscraper.xrxiv.xrxiv_api import MedRxivApi

from arxiv_paper_fetcher import remove_latex_tags

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == '__main__':
    begin_date = "2024-11-01"
    end_date = "2024-11-29"
    api = MedRxivApi(max_retries=10)
    csv_path = pathlib.Path("medrxiv_abstracts.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Abstract", "Link"])
        doi_numbers = set()
        paper_counter = 0

        for paper in api.get_papers(begin_date=begin_date, end_date=end_date):
            if paper["doi"] in doi_numbers:
                continue

            doi_numbers.add(paper["doi"])
            parsed_abstract = remove_latex_tags(paper["abstract"])
            writer.writerow([parsed_abstract, paper["doi"]])
            paper_counter += 1
            if paper_counter % 100 == 0:
                logging.info(f"Downloaded {paper_counter} articles.")

    logging.info(f"Fetching done downloading {paper_counter} articles.")
