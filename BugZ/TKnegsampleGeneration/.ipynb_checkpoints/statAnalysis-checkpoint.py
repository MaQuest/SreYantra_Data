'''
this program will used to analyze the statistics of the different data sets That
we have produced

Will be looking at statistics Regarding

Total Amount of Data (Dependent and Independent)

Amount of blocked requirements
Amount of "depends on" requirements
Amount of "duplicate" requirements

Amount of Pairs (Dependent and Independent)

Amount of Blocked Pairs (Dependent) where B requires A
Amount of "Depends On Pairs" (Dependent) where A requires # BUG:
Amount of Duplicates (Dependent) where A and B are similar

Looking at certain features of pairs
'''

import pandas as pd
import numpy as np


DEPENDENT_DATA = "AllData.csv"
INDEPENDENT_DATA = "AllIndependentData.csv"

DEPENDENT_PAIRS = "Level1PairsReadyData.csv"
#INDEPENDENT_PAIRS =

'''
function that will compute some stats based on inputs
'''

def analysis():
    df_d = pd.read_csv(DEPENDENT_DATA)
    df_i = pd.read_csv(INDEPENDENT_DATA)

    df_dp = pd.read_csv(DEPENDENT_PAIRS)

    print("Total amount of data points for dependent requirements: " + str(len(df_d)))
    print("Total amount of data points for indepedent requirements: " + str(len(df_i)))
    print("Total amount of data points for dependent pairs: " + str(len(df_dp)))
    print("Total amount of \'blocks\' data points: " + str(len(df_d[df_d["blocks"].notnull()])))
    print("Total amount of \'depends on\' data points: " + str(len(df_d[df_d["depends_on"].notnull()])))
    print("Total amount of \'duplicate data\' points: " + str(len(df_d[df_d["duplicates"].notnull()])))
    #print("Total amount of data points for blocked requirements: " + d




def main():
    analysis()

main()
