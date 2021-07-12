def get_final_txt(grb, tables, sentences, output_path):
    """
    Combine the data from [grb]_final_sentences.txt and [grb]_final_tables.txt.
    If a piece of data in tables and another piece in sentecnes are originially
    from the same GCN. Put them in the same GCN in [grb]_final.txt.
    """

    # Avoid modifying the data for the later use.
    tables = tables.copy()
    sentences = sentences.copy()

    # Open up the file.
    file = open(f"{output_path}{grb}/{grb}_final.txt", 'w')

    # Loop through the sentences and for each sentence, check if there is any table
    # that are originially from the same GCN.
    for sentence in sentences:

        # The number of the GCN.
        num = sentence['number']

        # The final string that we dumps into the text file.
        result = "=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n\n"
        result += f"GCN Number: {sentence['number']}\n\n"
        result += f"SENTENCE DATA:\n\n{sentence['sentences']}\n\n"

        # The variable to help check how many tables are from the same GCN.
        table_with_the_same_number = 0

        # Loop through the tables to see if there are any tables in the same GCN.
        for idx, table in enumerate(tables):
            
            # If we find any tables in the same GCN.
            if table['number'] == num:
                
                if table_with_the_same_number == 0:
                    result += "TABLE DATA:\n\n"

                table_with_the_same_number += 1
                result += '\n'.join(table['table']) + '\n\n'
                tables.pop(idx)

        file.write(result)

    # Write the remaining tables to the text file.
    for table in tables:

        result = "=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n\n"
        result += f"GCN Number: {table['number']}\n"
        result += "TABLE DATA:\n\n" + '\n'.join(table['table']) + '\n\n'

        file.write(result)
