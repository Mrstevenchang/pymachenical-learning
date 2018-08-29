import numpy as np
import math
def loadDataSet():
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]
	return postingList,classVec
	
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)
	
def setOfWords2Vec(vocabList,inputSet):
	returnVec = np.zeros(len(vocabList))
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print('the word :%s is not in my Vocabulary!'% word)
	return returnVec

def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	numWord = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/len(trainCategory)
	p0Num = np.zeros(numWord)
	p1Num = np.zeros(numWord)
	p0Demon = 0
	p1Demon = 0
	for i in range(numTrainDocs):
		if trainCategory[i] == 0:    #原始数据转化向量后中没有出现侮辱性词汇
			p0Num += trainMatrix[i]  ##在处理过的词汇表中每个词出现的
			p0Demon += sum(trainMatrix[i])
		else:						 #出现侮辱性词汇
			p1Num += trainMatrix[i]
			p1Demon += sum(trainMatrix[i])
	print(p0Num)
	print(p1Num)
	print(p0Demon)
	print(p1Demon)
	p0Vec = p0Num / p0Demon
	p1Vec = p1Num / p1Demon
	return p0Vec,p1Vec,pAbusive

def trainNB1(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	numWord = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/len(trainCategory)
	p0Num = np.ones(numWord)
	p1Num = np.ones(numWord)
	p0Demon = 2
	p1Demon = 2
	for i in range(numTrainDocs):
		if trainCategory[i] == 0:
			p0Num += trainMatrix[i]
			p0Demon += sum(trainMatrix[i])
		else:
			p1Num += trainMatrix[i]
			p1Demon += sum(trainMatrix[i])
	p0Vec = np.log(p0Num / p0Demon)
	p1Vec = np.log(p1Num / p1Demon)
	return p0Vec,p1Vec,pAbusive
	
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
	p0 = sum(vec2Classify*p0Vec) + np.log(1-pClass1)
	if p1>p0:
		return 1
	else:
		return 0
		
def testingNB():
	listPosts,listClasses = loadDataSet()
	myVocabList = createVocabList(listPosts)
	trainMat = []
	for postinDoc in listPosts:
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
	p0V,p1V,pAb = trainNB1(trainMat,listClasses)
	testEntry = ['love','my','dalmation']
	thisDoc = setOfWords2Vec(myVocabList,testEntry)
	print(testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb))
	testEntry = ['garbage','stupid']
	thisDoc = setOfWords2Vec(myVocabList,testEntry)
	print(testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb))