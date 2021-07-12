import os
from .constants import sentence_check


def check_sentence(grb_listing):
    """
    Check if there are sentences that include possible data points in grb_listing.
    """
    return sentence_check.search(grb_listing) != None


def get_final_sentences_txt(grb, output_path):
    """
    Fetch the data from [grb]_sentences.txt and select only the paragraphs
    with the possible data points identified by the regex, sentence_check.
    """

    # Get the lines of the text file.
    lines = open(f"{output_path}{grb}/{grb}_sentences.txt", 'r').read()

    GCNs = lines.split('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
    GCNs = [GCN.split('\n\n') for GCN in GCNs]

    # we now have a list of GCNs, where each GCN is a list of paragraphs.
    all_data = []
    data = {}

    # PHASE 1: Loop through the paragraphs and store the sentences with data points that we are interested in to all_data.
    for GCN_paragraphs in GCNs:

        for paragraph in GCN_paragraphs: 

            lines = paragraph.split('\n')

            for line in lines:

                # If line contains "NUMBER: "
                if "NUMBER: " in line:

                    # Add data only if it is not an empty dictionary.
                    if data:
                        all_data.append(data)

                    data = {}
                    data["number"] = line.strip("NUMBERS: ")
                    data["sentences"] = ""
                    continue
                
                # If there is a sentence matched in the line, add the entire paragraph.
                match = sentence_check.search(line)
                if match:
                    data["sentences"] += paragraph + "\n"
                    break

    # PHASE 2: Write the data into *_final_sentences.txt.
    file = open(f"{output_path}{grb}/{grb}_final_sentences.txt", 'w')
    prev_num = 0

    for data in all_data:

        # If the previous table and the current table is in the same gcn,
        # do not print out the header again.
        if prev_num == data['number']:
            result = ""
        else:
            result = "=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n\n"
            result += f"Number: {data['number']}\n"

        result += data['sentences'] + "\n"
        prev_num = data['number']
        file.write(result)

    # close the *_final_sentences.txt and remove the original *_sentences.txt.
    file.close()
    os.remove(f"{output_path}{grb}/{grb}_sentences.txt")
    return all_data


def final_sentences_to_csv(grb, output_path):
    pass