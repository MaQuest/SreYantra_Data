''' To generate independent pairs: find all the pairs from df_pairs_dependent, 
read all the ids from df_root, read all ids from df_select then exclude the items that have  '''

#read all Level1PairsReadyData.csv data
#extract req1Id and req2Id and generate sets. Now generate all possible pairs with them

#repeat it for Level2PairsReadyData.csv data

import pandas as pd

def flattenData():
    df_data = pd.read_csv("Level1PairsReadyData.csv")
    #print(df_data.columns[5:10])
    #print(df_data.columns[10:16])
    rng1 = list(range(5,10))
    df_p1 = df_data.iloc[:, rng1]
    df_p1.rename(columns={'req1':'req', 'req1Id':'reqId', 'req1Priority': 'reqPriority','req1Severity':'reqSeverity','req1Type':'reqType'}, inplace=True)
    rng2 = list(range(10,15))
    df_p2 = df_data.iloc[:, rng2]
    df_p2.rename(columns={'req2':'req', 'req2Id':'reqId', 'req2Priority': 'reqPriority','req2Severity':'reqSeverity','req2Type':'reqType'}, inplace=True)

    df_long = df_p1.append(df_p2, sort=True)
    df_long.drop_duplicates(subset ="reqId", keep = 'first', inplace = True)
    #df_long.to_csv("dummy.csv")
    #print (rng)
    print(df_p1.head())
    print(df_p2.head())
    #print ((df_data.iloc[:, rng2]).head())
    ids1 = list(set(df_data['req1Id']))
    ids2 = list(set(df_data['req2Id']))
    ids = ids1+ids2
    print (len(list(set(ids))), len(ids1), len(ids2), len(df_data))
    return df_long, df_data

def genPairs(lst,df_data,df_long):
    pairsList = []
    count = 0
    flag = False
    limit = 100
    for ele1 in lst:
        #print
        ele1_index = df_long.index[df_long['reqId']==ele1] #lookup the item in df_long
        if not ele1_index.empty: #if list is not empty
            index1 = ele1_index[0] 
            req1 = df_long.iloc[index1]['req']
            req1Id = df_long.iloc[index1]['reqId']
            req1Priority = df_long.iloc[index1]['reqPriority']
            req1Severity = df_long.iloc[index1]['reqSeverity']
            req1Type = df_long.iloc[index1]['reqType']
            #print(req1)
            #print(req1Id)    
            #print(len(df_long))
            #input("hit enter to proceed")
            #print(df_long.iloc[11617])
            #print(df_long.iloc[11618])
        for ele2 in lst:
            ele2_index = df_long.index[df_long['reqId']==ele2] #lookup the item in df_long
            if (ele1 != ele2): #dont generate pair of the type (a,a)
                if  not (df_data[(df_data['req1Id']!=ele1) & (df_data['req2Id']!=ele2) ]).empty: #this pair is not present as a dependent pair then
                    if (not ele2_index.empty) and (ele2_index<len(df_long)): #if list is not empty
                        index2 = ele2_index[0]
                        #print (index2) 
                        req2 = df_long.iloc[index2]['req']
                        req2Id = df_long.iloc[index2]['reqId']
                        req2Priority = df_long.iloc[index2]['reqPriority']
                        req2Severity = df_long.iloc[index2]['reqSeverity']
                        req2Type = df_long.iloc[index2]['reqType']
                        pair = {'req1':req1,'req1Id':req1Id, 'req1Type':req1Type,'req1Priority':req1Priority, 'req1Severity':req1Severity,'req2':req2,'req2Id':req2Id,'req2Type':req2Type,'req2Priority':req2Priority, 'req2Severity':req2Severity,'CosSimilarity':0,'SemSimilarity':0,'BinaryClass':0,'MultiClass':-1} 
                        #print (str(pair).encode("utf-8"))
                        count = count +1       
                        pairsList.append(pair)
                        
                        if count == limit:
                            df_allPairs = pd.DataFrame(pairsList)
                            if flag == False:
                                with open('dummy.csv', 'a',encoding='utf-8') as f:
                                    df_allPairs.to_csv(f, encoding='utf-8')                        
                                flag = True
                                
                            else:
                                with open('dummy.csv', 'a', encoding='utf-8') as f:
                                    df_allPairs.to_csv(f, header=False, encoding='utf-8')                        

                            pairsList = []
                            count = 0
                    #else:
                        #print (ele2_index, len(df_long))
                else:
                    print("-----------this pair is--------------")
                    print(df_data[(df_data['req1Id']!=ele1) & (df_data['req2Id']!=ele2)])
                    #input("hit enter to proceed")
    #df_allPairs = pd.DataFrame(pairsList)

    return df_allPairs


def main():
    df_long, df_data = flattenData()    
    ids = list(df_long['reqId'])
    #print (ids)
    #df_allPairs = 
    genPairs(ids,df_data, df_long)    
    #df_allPairs.to_csv("Level1_negativePairs.csv")

main()
    