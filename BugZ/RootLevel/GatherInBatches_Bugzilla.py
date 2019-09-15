import pandas as pd
import BugzillaDataReading as BZ

INPUT = "./ListAllStage3IDS.txt"

def split():
    df_data = pd.read_csv(INPUT)
    print(len(df_data))
    #read the data using REST API in a batch of 10,000 and dump 
    print(df_data[1:6])
    df_data[1:10001].to_csv("part1Stage2.csv")
    df_data[10001:20001].to_csv("part2Stage2.csv")
    df_data[20001:30001].to_csv("part3Stage2.csv")
    df_data[30001:40001].to_csv("part4Stage2.csv")
    df_data[40001:50760].to_csv("part5Stage2.csv")

def convert(lst):
    lst = [str(a) for a in lst]
    lst = ', '.join(lst)
    return lst

def REST_read():
    #extract all the ids in a big list
    df_part = pd.read_csv('part5Stage2.csv')
    #print(df_part['id'].head())
    Str_ids = convert(df_part['id'])
    Lst = Str_ids.split(',')
    Lst = [ int(x) for x in Lst ]
    print(type(Lst))

    #now invoke the function to fetch data
    DataLevelOne = BZ.fetchAllrecs(Lst)
    df_PartOut = pd.DataFrame(DataLevelOne)
    df_PartOut.to_csv("Data_part5Stage2.csv")
    
    pass



def main():
    #split()
    REST_read()

main()
