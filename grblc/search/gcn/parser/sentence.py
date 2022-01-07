import os
from .constants import sentence_check, mag_check, flux_check


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
    
    # Look through only sentence GCN  ### Doesn't always catch sentences - Nicole, July 13
    #lines = open(f"{output_path}{grb}/{grb}_sentences.txt", 'r').read()

    # Look through all GCNs
    lines = open(f"{output_path}{grb}/{grb}_all_gcn.txt", 'r').read()
   
    GCNs = lines.split('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
    GCNs = [GCN.split('\n\n') for GCN in GCNs]

    # we now have a list of GCNs, where each GCN is a list of paragraphs.
    all_data = []
    data = {}
    
    mag_data = {}
    
 

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
                    
                       
                        
        # Matches magnitudes in sentences              
        for paragraph in GCN_paragraphs:
            
            magMatch = mag_check.findall(paragraph)
            fluxMatch = flux_check.findall(paragraph)
           
            if magMatch:
                
                for entry in magMatch:
                    
                    # Append matches to dictionary w/ gcn as key
                    if data['number'] in mag_data:
                        mag_data[data['number']].append(entry)
                    else: 
                        mag_data[data['number']] = [entry]
            print(fluxMatch)
      
    # Take the information from mag_data dictionary and append to a .txt file
    datMag = []
    mag_data_sort = []
    mag_from_sentences = open(f'{output_path}{grb}/{grb}_sentences_mag.txt','w+')
    mag_from_sentences.write(str('gcn')+str('\t')+str('mag')+str('\t')+str('mag_err')+str('\t')+str('band')+str('\n'))
    
    for key in mag_data:
       
        gcnNum = key
        
        for sublst in mag_data[key]:

            band = sublst[4]
            mag = sublst[5]
            mag_err = sublst[7]
            
            mag_from_sentences.write(str(gcnNum)+str('\t')+str(mag)+str('\t')+str(mag_err)+str('\t')+str(band)+str('\n'))
        
    mag_from_sentences.close()
  

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