import re
import os
from .constants import dash_line, table_entry_check, sentence_check


def check_table(grb_listing):
    """
    Return if there is any tables in grb_listing.
    """

    # Lines of the grb_listing
    lines = grb_listing.split('\n')
    lines = [line for line in lines if line and not line.isspace()] # filter empty lines

    # The bool for checking if previous line is also matched as a table entry
    prev = False
    
    for line in lines:

        # First, check if there is a dash line.
        # Usually when there is a dash line, there is a table.
        match = dash_line.search(line)
        if match:
            return True


        # Second, check if there is a group of at least two table entries.
        table_match = table_entry_check.search(line)
        sentence_match = sentence_check.search(line)
        duplicate = table_match and sentence_match

        if table_match and not duplicate and prev:
            return True

        prev = not duplicate
        
    return False


def get_final_tables_txt(grb, output_path):
    """
    Fetch the data from the generated text file, *_table.txt.
    Store gcn numbers, dates, and tables and write the data
    accordingly into *_final.txt.
    """

    # Get the lines of the text file.
    lines = open(f"{output_path}{grb}/{grb}_table.txt", 'r').read().split('\n')
    lines = [line for line in lines if line and not line.isspace()] # filter empty lines
    
    allData = []
    data = {}

    # PHASE 1: Loop through the lines and store the data we are interested in to allData.
    while lines:
        
        line = lines.pop(0)

        # If line contains "NUMBER: "
        if "NUMBER: " in line:
            num = line.strip("NUMBERS: ")
            continue

        # Check if the line is matched as a table entry, and if it is not matched as a sentence.
        table_entry_match = table_entry_check.search(line)

        table = []

        # If there is a match, keep matching until the table ends.
        while table_entry_match:
            table.append(re.split("\t|\s{2,}", line))

            # If the table entry is the end of the file, lines will be an empty list.
            # Make sure we are not poping from an empty list.
            if lines:
                line = lines.pop(0)

                # Check if the line is matched as a table entry, and if it is not matched as a sentence.
                table_entry_match = table_entry_check.search(line)
            
            # Break the while loop, if there is no line anymore.
            else:
                break

        # If there is a group of at least two table entries, we add data to allData.
        # The reason that we have to store num and date is that there might be two 
        # tables in a gcn.
        if len(table) >= 2:
            data["number"] = num
            data["table"] = list(map(lambda t: '\t'.join(t), table))
            allData.append(data)
            data = {}             

    # PHASE 2: Write the data into *_final_table.txt.
    file = open(f"{output_path}{grb}/{grb}_final_table.txt", 'w')
    prev_num = 0

    for data in allData:

        # If the previous table and the current table is in the same gcn,
        # do not print out the header again.
        if prev_num == data['number']:
            result = ""
        else:
            result = "=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n\n"
            result += f"Number: {data['number']}\n"

        result += '\n'.join(data['table']) + '\n\n'
        prev_num = data['number']
        file.write(result)

    # close the *_final_table.txt and remove the original *_table.txt.
    file.close()
    os.remove(f"{output_path}{grb}/{grb}_table.txt")
    return allData


def final_tables_to_csv(grb, output_path="output/"):
    pass