'''
Program which processes the JSON file from https://bugzilla.mozilla.org/query.cgi?format=advanced
website, i.e the data for mozilla applications which are of the type tasks and enhancements.
These have EMPTY depends_on, dupiclates and Blocks fields.

Idea: is to iterate through each one of these ID's and then download the data for each item again using
REST API call : https://bugzilla.mozilla.org/rest/bug/

Stage 1:
Input: TasksAndEnhancements.Json
Output : AllData.csv

Stage 2:
Now recursively upto depth 5 get the element information for each on in the depends_on, blocks
and duplicate of fields
Input: AllData.csv
Output: AllData.csv
'''

import json, requests, itertools
import pandas as pd

INPUT = "./negTasksAndEnhancements.json"
STAGE1_OUTPUT = "AllIndependentData.csv"

STAGE2_INPUT = "AllIndependentData.csv"
STAGE2_OUTPUT = "AllIndependentData.csv"


REST_API = "https://bugzilla.mozilla.org/rest/bug/" # 1575284
def ReadInput():
    with open(INPUT, 'r', encoding='utf8') as f:
        distros_dict = json.load(f)
    list_elmts = distros_dict['bugs']
    df_Data = pd.DataFrame(list_elmts)
    #input("hit enter to proceed")
    return df_Data

def fetchAllrecs(eleList):
    BigList = []
    count = 1
    for i in eleList:
        count = count +1
        inst = FetItemInfo(i)
        if inst!=0:
            BigList.append(inst)
            # if count == 50:
            #     break
            print(str(count) + "/" + str(len(eleList)))
    return BigList

def convert(lst):
    lst = [str(a) for a in lst]
    lst = ', '.join(lst)
    return Lst

def FetItemInfo(id):
    new_url = REST_API+str(id)
    try:
        resp = requests.get(new_url)
        ele = resp.json()
        if ele['bugs'][0]['type'] in ('task','enhancement'):
            #print("found a task/enhancement")
            #ele['bugs'][0]['depends_on'] = convert(ele['bugs'][0]['depends_on'])
            #ele['bugs'][0]['duplicates'] = convert(ele['bugs'][0]['duplicates'])
            #ele['bugs'][0]['blocks'] = convert(ele['bugs'][0]['blocks'])
            return ele['bugs'][0]
        else:
            return 0
    except Exception:
        return 0

def stage1():
    '''Stage 1 : first set of records'''
    df_data = ReadInput()
    print(df_data.id)

    AllData = fetchAllrecs(df_data.id)

    #print (AllData)
    df_AllData = pd.DataFrame(AllData)
    df_AllData.to_csv(STAGE1_OUTPUT)

    return df_AllData


def combineAllinBigList(dfObj):
    #print("inside combineBigList")
    #print(dfObj.head(40))
    dfObj = dfObj.fillna(0)
    dfObj =  [x for x in dfObj if x != 0]
    #print(dfObj)
    #Llist = [ [int(s.strip()) for s in l] for l in dfObj]
    Llist = []
    for l in dfObj:
        llist = []
        if l!='':
            l=l.split(',')
            for s in l:
                try:
                    s = int(s)
                except ValueError:
                    pass
        #print(l)
                llist.append(s)
        Llist.append(llist)

    Llist = sum(Llist, [])
    #print(Llist)
    return Llist

'''
for every element in the Lst
check if the entry already exists in the dataframe
If not then using REST API collect it and append
'''
def recursiveStage(Lst,Lst_Root_ids):
    #find intesection of the two lists and then remove them from Lst
    Lst_intersection = [value for value in Lst if value in Lst_Root_ids] #elements common to both lists
    Leftside_Lst_intersection = [value for value in Lst if value not in Lst_Root_ids] #elements which ignore common elements

    #print(len(Lst_intersection))
    print(len(Leftside_Lst_intersection))
    return list(set(Leftside_Lst_intersection))

def stage3():
    #extract level 2 data, this time make sure the extraction is done for task and enhancement type only.
    #1 read the files and create a big file
    df_all_data = pd.read_csv("Data_part1.csv")
    df_all_data = df_all_data.append(pd.read_csv("Data_part2.csv"))
    df_all_data = df_all_data.append(pd.read_csv("Data_part3.csv"))
    df_all_data = df_all_data.append(pd.read_csv("Data_part4.csv"))
    print(len(df_all_data))

    #extract the elements from dataframe that are either task or enhancement only.
    df_tasks = df_all_data.loc[df_all_data['type'] == 'task'] #,'priority']
    df_enhance = df_all_data.loc[df_all_data['type'] == 'enhancement']#,'type','priority']
    #print(len(df_tasks), len(df_enhance))
    #print(df_tasks.head(), df_enhance.head())

    #combine the dataframes
    df_tasks_enhance = df_enhance.append(df_tasks)
    print(len(df_tasks_enhance))

    stage2(df_tasks_enhance,"ListAllStage3IDS.txt")


    pass

def ChangeStringListstoInts(df_data, col):

    df_data[col] = df_data[col].values.tolist()
    for index, row in df_data.iterrows():
        #print("in here")
        ilist = []
        #print (type(row[col]))
        if pd.notnull(row[col]):
            #print(str(row[col]).split(','))
            for item in str(row[col]).split(','):
                ilist.append(int(float(item)))
            #print(ilist)
            #df_data.at[index,col] = ilist
            row[col] = ilist

    #print (df_data[col])
    return df_data

def stage2(df,output):
    #df_data = pd.read_csv("dummy.csv")
    df_data = df       #pd.read_csv(input)
    print(len(df_data))

    '''by default csv has these three columns as str, convert it to list of numbers in dataframe
    as I need to look it up to further extract the data recursively in 2/3 iterations
    '''
    df_data=ChangeStringListstoInts(df_data, 'depends_on')
    df_data=ChangeStringListstoInts(df_data, 'blocks')
    df_data=ChangeStringListstoInts(df_data, 'duplicates')

    #now combine these respective lists in indexes to form a big single list
    #print(df_data['depends_on'].head(50))

    depends_on_List = combineAllinBigList(df_data['depends_on'])

    blocks_List = combineAllinBigList(df_data['blocks'])
    duplicates_List = combineAllinBigList(df_data['duplicates'])

    print("-------------------$$$$--------------------------------------")
    #print(depends_on_List)
    print(len(depends_on_List),len(blocks_List), len(duplicates_List))
    '''once again using REST API accumulate the data for these ID's in three lists above
    Check that the item does not exists already to eliminate duplicate entries
    index the dataframe based on 'id' column for easier grep
    '''
    #----------------------------stats-------------------------------
    CombinedList = depends_on_List + blocks_List + duplicates_List
    print("---------------------------------------------------------")
    print(len(CombinedList), len(set(CombinedList)))
    #print(CombinedList)
    # RootList = df_data['id'].tolist()
    # #print (RootList, len(RootList), type(RootList))
    # CombinedList = CombinedList + RootList
    # print("---------------------------------------------------------")
    # print(len(CombinedList), len(set(CombinedList)))
    # #----------------------------------------------------------------

    #for every element in the CombinedList extract data and add to df_data
    # Lst_TobeFetched = recursiveStage(list(set(CombinedList)),df_data['id'].tolist())
    f = open(output, "w")
    CombinedList = list(set(CombinedList))
    for i in CombinedList:
        f.write(str(i)+'\n')
    f.close()


    #just like stage1 fetch all but just append to existing file instead
    # DataLevelOne = fetchAllrecs(Lst_TobeFetched)
    # df_AllData = pd.DataFrame(DataLevelOne)
    # df_AllData.to_csv("Stage2Data.csv")
    # #with open(STAGE2_OUTPUT, 'a', encoding='utf8') as f:
    #    df_AllData.to_csv(f)

    #df_data = pd.read_csv(STAGE2_INPUT)
    #print(len(df_data))

    pass


def main():
    df_data = pd.read_csv(STAGE2_INPUT)
    print(df_data["product"])

main()
