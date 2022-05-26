import json
import os
import re
import shutil
import tarfile

import glob2
import requests

from .parser import check_sentence
from .parser import check_table
from .parser import final_sentences_to_csv
from .parser import final_tables_to_csv
from .parser import get_final_sentences_txt
from .parser import get_final_tables_txt
from .parser import get_final_txt


class Scraper:

    # TODO: We will have to change output_path to a non-default argument at the end.
    #       (because I beleive the client will not want the output to be in the module.)
    def __init__(self, data_path="", output_path=""):
        """ """
        self.set_data_path(data_path)
        self.set_output_path(output_path)

    def set_output_path(self, path=""):
        """
        Set output_path for this scraper object.
        """

        # Set up new_output_path and there are three cases.
        # If no path is passed, we assume the output path will be the folder in the gcn_crawler
        if not path:
            new_output_path = "gcncc/output/"

        # If the path ends with a path separator.
        elif path.endswith(os.path.sep):
            new_output_path = path + "gcncc/output/"

        # If the path does not end with a path separator.
        else:
            new_output_path = path + os.path.sep + "gcncc/output/"

        # Move the old output folder to the new location.
        # Skip this part if there is no old output path.
        try:
            old_output_path = self.__output_path__
            if os.path.exists(old_output_path):
                shutil.move(old_output_path, new_output_path)
        except:
            pass

        # Update output path.
        self.__output_path__ = new_output_path

    def set_data_path(self, path=""):
        """
        Set data_path for this scraper object.
        """

        # Set up new_data_path and there are three cases.
        # If no path is passed, we assume the data path will be the folder in the gcn_crawler
        if not path:
            new_data_path = "gcncc/data/"

        # If the path ends with a path separator.
        elif path.endswith(os.path.sep):
            new_data_path = path + "gcncc/data/"

        # If the path does not end with a path separator.
        else:
            new_data_path = path + os.path.sep + "gcncc/data/"

        # Move the old output folder to the new location.
        # Skip this part if there is no old output path.
        try:
            old_data_path = self.__data_path__
            if os.path.exists(old_data_path):
                shutil.move(old_data_path, new_data_path)
        except:
            pass

        # Update data path.
        self.__data_path__ = new_data_path

    # TODO: check the usage of save argument
    def grb_circulars(self, save=True):
        """
        Downloads tar file from GCN website into a folder gcn3.
        """

        # Get the url and https request.
        url = "https://gcn.gsfc.nasa.gov/gcn3/all_gcn_circulars.tar.gz"
        response = requests.get(url, stream=True)

        # TODO: Understnand this and comment.
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=self.__data_path__)

        return file

    def load_gcn(self):
        """
        Creates a list of all gcn files in gcn3
        """
        gcns = glob2.glob(self.__data_path__ + "gcn3/" + "*.gcn3")
        return gcns

    # TODO: double check it is finding ALL of them there was one test case where previous function
    #       returned more searches than this function
    # TODO: Check the usage of threading and echo arguments.
    def scrape(self, threading=True, echo=None):
        """
        Search for a GRB in the file
        """

        self.grb_circulars()
        error = 0
        circ_dict = {}
        fileErr = []
        gcns = self.load_gcn()

        # TODO: Understand this and comment.
        for fileName in gcns:

            file = open(fileName, encoding="utf8", errors="ignore")
            file_search = file.read()

            # Search the whole text for any instance of a GRB
            matches = re.findall("GRB\s?\d{6}[A-Z]?(?!\d)", file_search, flags=re.IGNORECASE)
            matches = [l.strip("GRBSWgrbsw\n ") for l in matches]

            #  TODO: Understnand this and comment.
            circ_dict[fileName.rsplit("/", 1)[-1]] = matches
            file.close()

        # Show if there are any problems while scraping.
        print("There were " + str(error) + " error(s) in the following file(s):")
        print(fileErr)

        # Write the dictionary to and output location
        with open(self.__data_path__ + "gcn_archive_circ_dict.json", "w") as f:
            f.write(json.dumps(circ_dict))

    # Function that searches dictionary for GRB (Add keywords like 'subaru' later)
    def grb_lookup(self, grb, *keywords):
        """
        Look up a specific grb in a circular dictionary.
        Look for specific keywords in the gcn if the client provides ones.
        """

        # Modify GRB if the length of it is smaller than 6
        if len(grb) < 6:
            grb = "0" * (6 - len(grb)) + grb

        # Check if output path already exists.
        if not os.path.exists(self.__output_path__):
            os.mkdir(self.__output_path__)

        # Check if output path of this specific GRB already exists.
        if not os.path.exists(f"{self.__output_path__}{grb}/"):
            os.mkdir(f"{self.__output_path__}{grb}/")

        # Make sure keywords is a tuple or None.
        assert not keywords or isinstance(keywords, tuple)

        # Load the data.
        try:
            filepath = self.__data_path__ + "gcn_archive_circ_dict.json"
            with open(filepath) as file:
                circ_dict = json.load(file)
                file.close()
        except FileNotFoundError:
            raise FileNotFoundError('The "data" folder could be removed. Try scrape().')

        # Get the list of gcn that has the grb.
        gcn_list = [key for key, list_of_grbs in circ_dict.items() if grb in list_of_grbs]

        # The json object to store types, files, and check functions to reduce repetition
        categories = [
            {"name": "all_gcn", "counter": 0, "file_name": f"{grb}_all_gcn.txt", "check": lambda x: True},
            {"name": "table", "counter": 0, "file_name": f"{grb}_table.txt", "check": check_table},
            {"name": "sentence", "counter": 0, "file_name": f"{grb}_sentences.txt", "check": check_sentence},
        ]

        # Open the files and store them.
        for cat in categories:
            cat["file"] = open(f"{self.__output_path__}{grb}/{cat['file_name']}", "w")

        for gcn in gcn_list:

            # Open file in the list of GCNs and copy text to new txt file.
            grb_open = open(
                self.__data_path__ + "gcn3/" + gcn, "r", errors="ignore"
            )  # Added the ignore attribute or it will raise an UnicodeDecodeError
            grb_listing = grb_open.read()

            # Loop through categories and use the check function in each category to filter GCNs
            for cat in categories:
                if cat["check"](grb_listing):
                    cat["file"].write(
                        f"=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n\n{grb_listing}\n"
                    )
                    cat["counter"] += 1

            # Check if any keyword is in the text.
            for k in keywords:

                key_check = re.search(k, grb_listing, flags=re.IGNORECASE)

                # If a key is matched, stop looping through the keywords and write the text into the text file.
                if key_check:
                    file = open(f"{self.__output_path__}{grb}/{grb}_{k}.txt", "w")
                    file.write(
                        "=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n\n%s\n" % grb_listing
                    )

                    categories[0]["counter"] += 1
                    print(f"{categories[0]['counter']}. {gcn} for {k}\n")
                    file.close()

        # Close the files and report the results.
        for cat in categories:
            cat["file"].close()

        # Get the data object from get_final_*_txt functions
        allData_tables = get_final_tables_txt(grb, self.__output_path__)
        allData_sentences = get_final_sentences_txt(grb, self.__output_path__)

        # Get the *_final.txt
        get_final_txt(grb, allData_tables, allData_sentences, self.__output_path__)
        return

    # We will move these functions back to grb_lookup after they are finished.
    def final_tables_to_csv(self, grb):
        final_tables_to_csv(grb, output_path=self.__output_path__)

    def final_sentences_to_csv(self, grb):
        final_sentences_to_csv(grb, output_path=self.__output_path__)
