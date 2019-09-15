'''
This program is to do the following
-- Read the 4 files in a single dataframe
-- Check if they are all aligned
-- in a loop only extract the samples which have depends_on and blocks on type="enhancement" Or "task" 
i.e ignore all the bugs.
-- for every pair create a dependency pair with following fields
req1, req2, req1_id, req2_id, req1_priority, req2_priority, cosise_similarity, semantic similarity, BinaryClass, MultiClass fields

Logic:
Step1: #dependent pairs (positive samples)
For every row: req1 in the dataframe
    extract the dependes_on and blocks fields
     for every entry: req2 
        if it is of the type "enhancement/task"
            create a pair req1, req2 and add to new dataframe

Step2: Independent pairs (negative samples)
For all the pairs which do not exists in the step1 do this comparison based on ids
create pairs and add them as independent pairs in dataframe.
'''
import pandas as pd


def readCombineSelect():

    df_all_data = pd.read_csv("Data_part1Stage2.csv")
    df_all_data = df_all_data.append(pd.read_csv("Data_part2Stage2.csv"),sort=True)
    df_all_data = df_all_data.append(pd.read_csv("Data_part3Stage2.csv"),sort=True)
    df_all_data = df_all_data.append(pd.read_csv("Data_part4Stage2.csv"),sort=True)
    df_all_data = df_all_data.append(pd.read_csv("Data_part5Stage2.csv"),sort=True)
    print(len(df_all_data))
    
    #df_select holds the task/enhancements only across all the links that root data holds. 
    df_select = df_all_data[df_all_data['type'].isin(['task','enhancement'])]
    print(len(df_select))
    #print(df_select.groupby(['product']).groups) #.value_counts())
    #print(df_select.groupby('product')['id'].value_counts())
    #print(df_select[['product']].count)
    #print(df_select.groupby('product').count())
    #stats = df_select.groupby('product')['id'].count()
    #print(type(stats))
    #stats.to_csv("statsStage2.csv")
    
    #daat.YEARMONTH.value_counts()
    #Create a network to generated dependency graph
    #print(df_select.iloc[1]['summary'])
    #print(df_select.iloc[1]['type'])
    #print(df_select.iloc[1]['id'])
    #input("hit enter to proceed")
    return df_select

def readRootData():
    # df_root = pd.read_csv("AllData.csv")
    # print("Size of root records is: %d" %(len(df_root)))
    df_root = pd.read_csv("../level 1 data/Data_part1.csv")
    df_root = df_root.append(pd.read_csv("../level 1 data/Data_part2.csv"),sort=True)
    df_root = df_root.append(pd.read_csv("../level 1 data/Data_part3.csv"),sort=True)
    df_root = df_root.append(pd.read_csv("../level 1 data/Data_part4.csv"),sort=True)
    df_root = df_root[df_root['type'].isin(['task','enhancement'])]
    print(len(df_root))
    return df_root

def ChangeStringListstoInts(Lst):
    ilist = []
    # print(Lst, type(Lst))
    # input("hit enter to proceed")
    if Lst!='nan':
        Lst = str(Lst).split(',')
        if Lst: #if list not empty
            for i in Lst:
            #print(str(row[col]).split(','))
                ilist.append(int(float(i)))
        #print(ilist)
    return ilist
    
def searchAndpair(lst,item,df_select,multiclass):
    #df_temp_pairs = pd.DataFrame(columns=['req1','req1Id','req1Priority', 'req1Severity','req2','req2Id','req2Priority', 'req2Severity','CosSimilarity','SemSimilarity','BinaryClass','MultiClass'])
    req1 = item['summary']
    req1Id = item['id']
    req1Priority = item['priority']
    req1Severity = item['severity']
    req1Type = item['type']
    req1Version = item['version']
    req1Release = item['target_milestone']
    req1Product = item['product']
    temp_pairs = []
    lst = ChangeStringListstoInts(lst) #its a bug string eg: '1232323,454545343,3434343'
    print(lst)
    for i in lst:
        #Following line returns a list
        print (i)
        instance = df_select.loc[df_select['id']==i]
        if not instance.empty:
            #indexIs = ele_index[0] 
            #print (instance)
            req2 = instance.iloc[0]['summary']
            req2Id = instance.iloc[0]['id']
            req2Priority = instance.iloc[0]['priority']
            req2Severity = instance.iloc[0]['severity']
            req2Type = instance.iloc[0]['type']
            req2Version = instance.iloc[0]['version']
            req2Release = instance.iloc[0]['target_milestone']
            req2Product = instance.iloc[0]['product']
            # print(req2, req2Id)
            # input("hit enter to proceed")    
        
            if multiclass =='depends_on': #it is "a requires b"  
                pair = {'req1':req1,'req1Id':req1Id, 'req1Type':req1Type,'req1Priority':req1Priority, 'req1Severity':req1Severity,'req1Ver':req1Version,'req1Release': req1Release, 'req1Product':req1Product, 'req2':req2,'req2Id':req2Id,'req2Type':req2Type,'req2Priority':req2Priority, 'req2Severity':req2Severity,'req2Ver':req2Version, 'req2Release': req2Release, 'req2Product':req2Product,'CosSimilarity':0,'SemSimilarity':0,'BinaryClass':1,'MultiClass':'requires'}
            elif multiclass == 'blocks': #it is "b requires a"
                pair = {'req1':req2,'req1Id':req2Id,'req1Type':req2Type,'req1Priority':req2Priority, 'req1Severity':req2Severity, 'req1Ver':req2Version, 'req1Release': req2Release, 'req1Product':req2Product, 'req2':req1,'req2Id':req1Id,'req2Type':req1Type,'req2Priority':req1Priority, 'req2Severity':req1Severity,'req2Ver':req1Version, 'req2Release': req1Release, 'req2Product':req1Product,'CosSimilarity':0,'SemSimilarity':0,'BinaryClass':1,'MultiClass':'requires'}
            else: #it is duplicate type
                pair = {'req1':req1,'req1Id':req1Id,'req1Type':req1Type,'req1Priority':req1Priority, 'req1Severity':req1Severity, 'req1Ver':req1Version, 'req1Release': req1Release, 'req1Product':req1Product,'req2':req2,'req2Id':req2Id,'req2Type':req2Type,'req2Priority':req2Priority, 'req2Severity':req2Severity, 'req2Ver':req2Version, 'req2Release': req2Release, 'req2Product':req2Product,'CosSimilarity':0,'SemSimilarity':0,'BinaryClass':1,'MultiClass':'similar'}
            
            #print(pair)
            print("---------------------------------------------------------------------")
            temp_pairs.append(pair)
        #break
        #print (df_temp_pairs)
    df_temp_pairs = pd.DataFrame(temp_pairs)
    return df_temp_pairs


def main():
    #1 read the files and create a big file
    df_select = readCombineSelect()
    df_root = readRootData()

    input("hit enter to proceed")
    #2 Step1: Dependent pairs
    '''
    Read the Root dataframe (approx 4000 records) in to dataframe
    For every item, for every entry in depends_on, blocks and duplicate, search the item in df_select
    '''
    df_pairs_dependent = pd.DataFrame(columns=['req1','req1Id','req1Type','req1Priority', 'req1Severity','req2','req2Id','req2Type','req2Priority', 'req2Severity','CosSimilarity','SemSimilarity','BinaryClass','MultiClass'])
    for index,ele in df_root.iterrows():
        lst_depends_on = ele['depends_on']
        lst_blocks = ele['blocks']
        lst_duplicate = ele['duplicates']
        df_pairs_dependent = df_pairs_dependent.append(searchAndpair(str(lst_depends_on),ele,df_select,'depends_on'))
        df_pairs_dependent = df_pairs_dependent.append(searchAndpair(str(lst_blocks), ele,df_select,'blocks'))
        df_pairs_dependent = df_pairs_dependent.append(searchAndpair(str(lst_duplicate), ele,df_select,'duplicates'))

    print("-------------------------End--------------------------")
    print(len(df_pairs_dependent))
    df_pairs_dependent.to_csv("Release_Level2PairsReadyData.csv")
    #3 Step2: Independent pairs
    '''
    To generate independent pairs: find all the pairs from df_pairs_dependent, 
    read all the ids from df_root, read all ids from df_select then exclude the items that have dependent pairs.
    kind of left out intersection
    '''
    #have written a separate program for this. under negativesamplegeneration folder
    pass

main()