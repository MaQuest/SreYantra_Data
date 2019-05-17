import textwrap
import re
from nltk.tokenize import sent_tokenize, word_tokenize, regexp_tokenize
import csv
import numpy as np
import pandas as pd
import sys, os

######################################################################################
######################################################################################
######################################################################################
#
# This program takes a requirment document in heir format, creates a dictionary data structure
# and writes it to CSV. Hardcoded file names:
#
# 	csv_file = "ef7.csv"
# 	file = open("clean - 2007 - eirene fun 7.txt",mode='r')
# 	fout = open("out.txt",mode='w')
# 	fout = open("subSec.txt",mode='w')
#
######################################################################################
######################################################################################
######################################################################################
def sysIO():
	specialChar = r"\."
	if(len(sys.argv)==1):
		inputFile = input('please enter name of file you would like to parse followed by .txt')
		if (os.path.isfile(inputFile)==False):
			print('invalid file name')
			sys.exit(1)
		outFile=input('please enter what file name you would like to be created')
		c='.csv'
		outFile=outFile +c
	elif(len(sys.argv)==2):
		if(re.search(specialChar, sys.argv[1])):
			inputFile=sys.argv[1]
			if (os.path.isfile(inputFile)==False):
				print('invalid file name')
				sys.exit(1)
		else:
			outFile = sys.argv[1]
			inputFile = input('please enter name of file you would like to parse followed by .txt')
			if (os.path.isfile(inputFile)==False):
				print('invalid file name')
				sys.exit(1)
	elif(len(sys.argv)==3):
		i=1
		j={1,2}
		while i < 3:
			if(re.search(specialChar, sys.argv[i])):
				inputFile=sys.argv[i]
				if (os.path.isfile(inputFile)==False):
					print('invalid file name')
					sys.exit(1)
				b={i}
				j-=b
				for item in j:
					outFile=sys.argv[item]
			i+=1
	else:
		print('invalid arguments, please retry with proper arguments')
		sys.exit(1)       
	return inputFile, outFile 
######################################################################################
######################################################################################
######################################################################################
#
# This function takes a requirment document in txt format. For each line of the file
# write the line to another txt file if the data is not metadata, or non pertinent
# information
#
######################################################################################
######################################################################################
######################################################################################

def cleanDoc(inFile):

	file = open(inFile,mode='r')
	fout = open("out.txt",mode='w')

	contents = file.readlines()
# Character symbol for top of page
	specialChar = r"\x0c"
# Character symbol for PSA page indicator
	pageInd = r"^PSA|^Page"
# Character symbol for the misceleneous references at end of doc
	refId = r"^[0-9]\s\‘"
	for line in contents:
		newLine=line
		if (re.match(specialChar, newLine)):
			fout.write('')
			# print('non ascii count',count)
		elif(re.match(pageInd, newLine)):
			fout.write('')
		elif(re.match(refId, newLine)):
			fout.write('..................')
			break
			# print("references", newLine)
		else:
			fout.write(newLine)

	fout.close()
	file.close()

######################################################################################
######################################################################################
######################################################################################
#
# This function takes a requirment document in txt format. For each line of the file
# write the line to another txt file if the data is not metadata, or non pertinent
# information. The it creates a dictionary containing all requirements (also a dictionary)
# and the details of the requirments. Returns the dictionary
#
######################################################################################
######################################################################################
######################################################################################

def subSec():
	file = open("out.txt",mode='r')
	fout = open("subSec.txt",mode='w')
	contents = file.readlines()
	count = 0
	lineNum=0
	totLine=0
	num=0
	allSubSecs = {}
# requirment ID identifier regex 
	subSecInd = r"^[0-9]([0-9])*(.[0-9])*"
# chapter identifier regex
	chapInd = r"^[0-9]([0-9])*\s"
# start of requirement identifer regex
	strtInd = r"(^1.1 General)"
# regex to identify if the first character found on the is a number but not a req ID
	rNumId= r"^[0-9]\S*\)"
	newLn = r"$/n"
# Find start of the requirments by finding the second 1.1 General line (first is in glossary)
# Tracks the total amount of lines in the file 
	for line in contents:
		if (num==2):
			lineNum=count-1
			num+=1
		if (re.match(strtInd, line)):
			num+=1
		count+=1
	totLine=count-1
# this writes a new document, with just the pertinent sections to be processed. Saved in subSec.txt
	count=0
	for line in contents:
		if(count>=lineNum):
			line.rstrip(' \n')
			fout.write(line)
		count+=1

	fout.close()
# This loops through each line of the file. if the first word of line fits into the req ID regex,
# not a chapter head and not a random number within the details of a req ID, then save it as a 
# dictionary containing the ID and the remaining portion on the line. To get the remaining portion 
# of details not contained in the first loop, it loops the next lines until another ID is found, 
# concatinating the first line with the rest. Once complete, add it to a dictionary containing 
# all of the subsections
	fin = open("subSec.txt",mode='r')
	contents = fin.readlines()
	count=0
	i=0
	j=0
	for line in contents:
		count+=1
		line=line.rstrip()
		line=re.sub(newLn,'', line)
	while i < count:
		if (re.match(subSecInd, contents[i])and not re.match(chapInd, contents[i]) and not re.match(rNumId, contents[i]) ):
			words=contents[i].split(" ",1)

			newRQ ={}
			newLabel = words[1]
			newLabel=newLabel.rstrip(' \n')
			newLabel=newLabel.strip('\n')
			newRQ['SS']=words[0]

			while(not re.match(subSecInd, contents[i+1]) and i+1<count-1):
				addLine=contents[i+1]
				addLine =addLine.rstrip(' \n')
				addLine = addLine.lstrip('\n')
				newLabel+=addLine
				# print(i)
				i+=1
			newRQ['label']=newLabel
			allSubSecs['SubSec '+words[0]]=newRQ
			# print(newRQ)
		i+=1

	fin.close()
	return allSubSecs
# Save dic to data frame, write to CSV
def pandaDtoC(subSecDic, outFile):
	csv_file = outFile
	newDF= pd.DataFrame.from_dict({(i): subSecDic[i]
		for i in subSecDic.keys()
		for j in subSecDic[i].keys()
		for j in subSecDic[i].keys()}, orient='index')
	export_csv = newDF.to_csv(csv_file)
	print(newDF)


def main():
	iFile, oFile = sysIO()
	cleanDoc(iFile)
	data = subSec()
	pandaDtoC(data,oFile)


main()