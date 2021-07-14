'''
Test script that will read automatically read through GCNs and allow the user to accept or reject a data point given the point and paragraph/table

'''
import os

## Work in progrress


## Work in progrress

def manual_sort(filepath, grb):
    
    flagged_data = open(filepath+str(grb)+'_sentences_mag.txt','r')
    accepted_data = open(filepath+str(grb)+'_acceped_mag.txt', 'w')
    
    accepted_data.write(str('gcn')+str('\t')+str('mag')+str('\t')+str('mag_err')+str('\t')+str('band')+str('\n'))
    
    lines = flagged_data.readlines()[1:]
    
    
    
    for point in lines:
        
        file = 'C:\\Users\\nicol\\Desktop\\GitHub\\grb\\gcn_crawler\\gcncc\\data\\gcn3\\'
        gcn = point.split('\t')[0]
        print(gcn)
        
        file = file+str(gcn)+str('.gcn3')
        print(file)
        
        
        
        loopVal = True
        print('==========================\n')
        print(point)
        
        
        
        while loopVal:
            
            os.system(file)
            
            userInput = input('Accept the above point? y/n \n')
            
            
            if userInput == 'n':
                loopVal = False
                
        
            elif userInput == 'y':
                
                accepted_data.write(str(point)+str('\n'))
                
                loopVal = False

            else:
                print('Please input a valid response: (y or n)')
                pass
    flagged_data.close()
    accepted_data.close()
    



