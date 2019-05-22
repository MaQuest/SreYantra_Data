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
# This tool takes in two arguments on command line, identifies if the data is in heir or flat
# structure, and writes the requirments and descriptions to csv format 

# please have the input files saved in the same file you are running the tool from
#
######################################################################################
######################################################################################
######################################################################################


######################################################################################
######################################################################################
######################################################################################
# sysIO tests how many arguments the user gave the command line when starting the program.

# The assumptions are if the number of arguments the user gave is:
# 1 then Need an input and output file
# 2 then one file given, might be input or output
# 3 then two files given, one input, one output


# Then the program tests the efficacy of the input file given by seeing if its an actual pathway (.isfile())

# program returns all of the pertinent file names
######################################################################################
######################################################################################
######################################################################################


def sysIO():

# special character that identifies . in .txt which is used to see if the argument is input or output
	specialChar = r"\."
# string used to concat the given output file name with the string to get it in the right format 
	c='.csv'
# these arguments are based on the logic given within the subfunction description 
	if(len(sys.argv)==1):
		inputFile = input('please enter name of file you would like to parse followed by .txt')
		if (os.path.isfile(inputFile)==False):
			print('invalid file name')
			sys.exit(1)
		outFile=input('please enter what file name you would like to be created')
		outFile=outFile +c
	elif(len(sys.argv)==2):
		if(re.search(specialChar, sys.argv[1])):
			inputFile=sys.argv[1]
			if (os.path.isfile(inputFile)==False):
				print('invalid file name')
				sys.exit(1)
		else:
			outFile = sys.argv[1]
			outFile=outFile +c
			inputFile = input('please enter name of file you would like to parse followed by .txt')
			if (os.path.isfile(inputFile)==False):
				print('invalid file name')
				sys.exit(1)

# this portion has to creat a set of the command line indexes, tests one to see which format,
# if it is the input format, remove the index from the set, and set the other to outfile
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
					outFile=outFile +c
					
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
# the following lines innitiate the line number indictors
# count refers to the number of total lines in the pertinent file
	count = 0
# lineNum indicates which line number the requirements start on 
	lineNum=0
# totLine gives the index to the final line in file 
	totLine=0
# num is a tool used to check where the start of the requirements are.
	num=0
	allSubSecs = []
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
# this writes a new document, starting at the linNum(start or requirements), with just the pertinent sections to be processed. Saved in subSec.txt
# count gives the total number of lines in the newly created document
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
# concatinating the first line with the rest. Once complete, appends it to the array containing 
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
			allSubSecs.append(newRQ)
		i+=1
	# print(allSubSecs)
	fin.close()
	
	return allSubSecs            
            
            
# this function takes the string entered for output.csv on command line, saves the heir or flat data structure to a dataframe
# checks the status (flat or heir), saves it to the according csv format 
def pandaDtoC(subSecDic, outFile, status):
	csv_file = outFile
	dataStructure = pd.DataFrame(subSecDic)
	print(dataStructure)
	if status == True:
		export_to_csv = dataStructure.to_csv(csv_file, index=None, header=True, columns=["SS", "label"],encoding='utf-8-sig')
	else:
		export_to_csv = dataStructure.to_csv(csv_file, index=None, header=True, columns=["requirement", "description", "details"],encoding='utf-8-sig')

#columns=["requirement", "description", "details"],
#this function identifies whether the file passed in is in heir or flat structure
# does this by the logic that heir data structure will have #.#. requirement IDs vs flat 
# which max out at #. 
# returns True for heir and false for flat 
def heirOrFlat(inputFile):

	file = open(inputFile,mode='r')
	contents = file.readlines()
	heirData = r"^\d\.\d\."
	status=False
	for line in contents:	
		if (re.match(heirData, line)):
			status = True
			break
		else:
			status = False
	file.close()
	return status

# This function uses a modifed version of  Carmens code 
# takes the input file name as input and returns the data structure of an array containing a dictionary with keys [requirement],[descriptions], [details]
def flatStructure(inputFile):
	input_file = inputFile

	# opens input file in write mode, ignores errors
	with open(input_file, mode ='r', errors='ignore') as f:
		content = f.readlines()

# an array containing dictionaries of requirment, description and details 
	complete_list = [] # requirements list
	csv_columns = ["requirement","description","details"] # top row as column names
	heading = " "
	heading_number = "0" # so that heading != heading_number before reading the first line
	new_row = {}

	for line_count in range(len(content)): # for each line count in the length of the content
		if re.match("^(?:[0-9]\.[0-9]+)+",content[line_count]): # lines that begin with decimal between 2 numerical values (eg. 1.2)
			if heading in heading_number: #only append details if heading matches heading_number (to prevent errors in first heading of doc)
				complete_list.append(new_row)
			new_row = {}
			word_list = content[line_count].strip().split(" ") # list of words of a line, including the numbers. strips the spaces at both ends and splits by space
			heading_number = word_list[0] # heading numbers
			heading = heading_number
			desc = " ".join(word_list[1:]).lstrip() # removes heading number, joins text in line by space, strips spaces at beginning of string
			desc = desc.lstrip('\n')
			desc = desc.rstrip('\n')
			new_row["requirement"]=heading_number
			new_row["description"]=desc	
		
		if re.match("^[^0-9]",content[line_count]): # if line does not begin with a number
			if "details" in new_row:
				cleanContent = content[line_count].replace('\n', ' ')
				cleanContent = cleanContent.strip()
				cleanContent=cleanContent.lstrip('\n')
				new_row['details']=new_row['details']+cleanContent # if details already exist, append next line to details key
			else:
				cleanContent = content[line_count].strip()
				new_row['details']=cleanContent # if details is not created, create details key and add content in value
		
		else:
			continue
	complete_list.append(new_row) # appends last row to complete_list

	return complete_list


# the main function takes in command line arguments for file identification, tests weather the pertinent file is in heir or flat formart
# uses the previously explained functions to parse the file into the pertinent datastructure, writes the data structure to csv format 
def main():

	iFile, oFile = sysIO()
	status = heirOrFlat(iFile)
	if status ==True:
		cleanDoc(iFile)
		data = subSec()
		pandaDtoC(data,oFile,status)
	else:
		data=flatStructure(iFile)
		pandaDtoC(data,oFile,status)


main()