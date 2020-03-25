
import os
import pickle
import sys
import numpy as np

import AST.AST as AST
from AST.PrintAST import PrintAST
from AST.MtdAST import MtdAST

import TF.Graph as Graph
from TF.TFNodesAST import TFNodesAST

def checkTFNodeNameForEq(curNodeOp:str, givenOp:str):
	return (curNodeOp == "\"" + givenOp + "\"")

def generateASTForNode(graph, curNode, dictNodeNameToOutVarStr, extraNodeInfoDict):
	# print("===>>> Generating AST for (nodeOp, nodeName) : (" + curNode.getOp() + ", " + curNode.getName() + ")")
	curNodeOp = curNode.getOp()
	ast = None
	func = getattr(TFNodesAST, curNodeOp[1:-1]) #To remove the " at the begin and end
	(assignedVarAST, curAST) = func(graph, curNode, dictNodeNameToOutVarStr, extraNodeInfoDict)
	return (assignedVarAST, curAST)

#Takes the graph DS and outputs IR in SeeDot for the same
def generateIRCode(graph, extraInfoDict):
	program = None
	innerMostLetASTNode = None
	dictNodeNameToOutVarStr = {}
	outVarCt = 0
	outVarPrefix = "J"
	mtdAST = MtdAST()
	for curNode in graph.getAllNodesRef():
		for curInp in curNode.getInputsRef():
			assert(curInp in dictNodeNameToOutVarStr) #Consequence of topological sorting of the TF graph

		if (curNode.getName() == "\"Placeholder\""):
			curOutVarStr = "X" 
			# first column corresponds to Y (output) 
			#lol = np.load('../model-processed/TF/bonsai/usps10/')[:,1:]
			values = np.load('../dataset-processed/TF/bonsai/usps10/train.npy')[:,1:].flatten()
			maxValue = -1e9
			minValue = 1e9
			for x in values:
				maxValue = max(maxValue, x)
				minValue = min(minValue, x)
			extraInfoDict[curNode.getName()]=extraInfoDict[curNode.getName()]+((minValue,maxValue),)
		# All hyperparameters are should be extracted from some file
		# currently hardcosing it for Bonsai
		elif (curNode.getOp() == "\"Placeholder\""):
			curOutVarStr = curNode.getName()[1:-1]
			value = np.load('../model-processed/TF/bonsai/usps10/' + curOutVarStr + '.npy').flatten()
			extraInfoDict[curNode.getName()]=extraInfoDict[curNode.getName()]+(value,)
		# All the trained variables are stored as npy files
		elif (curNode.getOp() == "\"VariableV2\""):
			curOutVarStr = curNode.getName()[1:-1]
			lol2 = np.load('../model-processed/TF/bonsai/usps10/' + curOutVarStr + '.npy')
			values = np.load('../model-processed/TF/bonsai/usps10/' + curOutVarStr + '.npy').flatten()
			maxValue = -1e9
			minValue = 1e9
			for x in values:
				maxValue = max(maxValue, x)
				minValue = min(minValue, x)
			extraInfoDict[curNode.getName()]=extraInfoDict[curNode.getName()]+((minValue,maxValue),)
			print ("Heyooo ", curOutVarStr, (minValue, maxValue), extraInfoDict[curNode.getName()])
		else:
			curOutVarStr = outVarPrefix + str(outVarCt)

		(assignedVarAST, curAst) = generateASTForNode(graph, curNode, dictNodeNameToOutVarStr, extraInfoDict)
		
		# This metadata information is used only for commenting 
		# TODO: Integrate this
		#mtdForCurAST = {AST.ASTNode.mtdKeyTFOpName : curNode.getOp()[1:-1],
		#				AST.ASTNode.mtdKeyTFNodeName : curNode.getName()[1:-1]}

		if (curAst is None):
			dictNodeNameToOutVarStr[curNode.getName()] = None
			continue



		print(curOutVarStr, curNode.getName())
		curOutVarAstNode = (assignedVarAST if assignedVarAST else AST.ID(curOutVarStr))
		if program:
			assert(type(innerMostLetASTNode) is AST.Let)
			newNode = AST.Let(curOutVarStr, curAst, curOutVarAstNode)
			#mtdAST.visit(newNode, mtdForCurAST)
			innerMostLetASTNode.expr = newNode
			innerMostLetASTNode = newNode
		else:
			innerMostLetASTNode = AST.Let(curOutVarStr, curAst, curOutVarAstNode)
			#mtdAST.visit(innerMostLetASTNode, mtdForCurAST)
			innerMostLetASTNode.depth = 0
			program = innerMostLetASTNode
		dictNodeNameToOutVarStr[curNode.getName()] = curOutVarStr
		outVarCt += 1
	return (program, dictNodeNameToOutVarStr)

def countUniqueOps(graph):
	allOps = []
	for curNode in graph.getAllNodesRef():
		if (curNode.getOp() not in allOps):
			allOps.append(curNode.getOp())
	print("allOps.ct = ", len(allOps))
	gradientDesOps = []
	for curNode in graph.getAllNodesRef():
		if (curNode.getName().startswith("\"gradient_descent_optimizer")) and (curNode.getOp() not in gradientDesOps): 
			gradientDesOps.append(curNode.getOp())
	print("allOps ct for gradient descent optimiser = ", len(gradientDesOps))
	return allOps

def readSizeInfo(fileName):
	allLines = None
	with open(fileName) as f:
		allLines = f.readlines()
	sizeInfo = {}
	for line in allLines:
		tokens = line.split()
		#assert(len(tokens) > 1) # Nodes with no size info are not getting outputted right now
		nodeName = tokens[0]
		tokens = tokens[1:]
		nodeOPSize = []
		for curDimStr in tokens:
				if (curDimStr == ''): continue
				nodeOPSize.append(int(curDimStr))
		sizeInfo[nodeName] = nodeOPSize
	return sizeInfo

# Since later on in the pipeline, the placeholder nodes which come up as cin statements
# 	are to be excluded from the timing calculation, output all such PlaceHolder nodes together first.
#	This doesn't violate the topological ordering because all such PlaceHolder nodes are leaf nodes 
# 	in the graph.
def prefixAllPlaceHolderNodes(graph):
	allNodes = graph.getAllNodesRef()
	placeHolderNodes = []
	remNodes = []
	for curNode in allNodes:
		if (curNode.getOp() == "\"Placeholder\"" or curNode.getOp() == "\"VariableV2\""):
			# Assert this is indeed a leaf node
			assert(len(curNode.getInputsRef()) == 0)
			placeHolderNodes.append(curNode)
		else:
			remNodes.append(curNode)
	graph.setNodesList(placeHolderNodes + remNodes)

def main():
	sys.setrecursionlimit(5000)

	print(os.getcwd())
	graphFileName = os.path.join('TF/TFData/Bonsai_usps10/graphDef.mtdata')
	graph = Graph.Graph()
	with open(graphFileName) as file:
		graph.readFromFilePointer(file)

	# # Read the sizeInfo also
	sizeInfoFileName = os.path.join('TF/TFData/Bonsai_usps10/sizeInfo.mtdata')
	sizeInfo = readSizeInfo(sizeInfoFileName)

	# Place all PlaceHolder nodes together at the beginning
	prefixAllPlaceHolderNodes(graph)

	# Re-format the input names of nodes
	for curNode in graph.getAllNodesRef():
		inputsRef = curNode.getInputsRef()
		for i,curInput in enumerate(inputsRef):
			#TODO for training : below is not correct
			# if (curInput.endswith(':1"')):
			# 	inputsRef[i] = curInput.split(':1')[0] + '"'
			if (curInput.startswith('"^')):
				# My hypothesis from empirical observation is that inputs which have '^' ahead of the node name
				#	denote control flow dependency and not data dependency.
				#	For all purposes for this compilation, control and data dependency is considered same.
				inputsRef[i] = '"' + curInput.split('^')[-1]

	# Create extra info dict
	# Format : (sizeInfo)
	extraInfoDict = {}
	for k,v in sizeInfo.items():
		print(k,v)
		extraInfoDict["\"" + k + "\""] = (v,)
	for curNode in graph.getAllNodesRef():
		if (curNode.getName() not in extraInfoDict):
			extraInfoDict[curNode.getName()] = (None,)
	
	print("Generating code from TF graph def : ", graphFileName, " ...")
	(program, dictNodeNameToOutVarStr) = generateIRCode(graph, extraInfoDict)
	printAST = PrintAST()
	printAST.visit(program)

	return program
	#print("SeeDot AST generation done. Pickling the AST.")
	#with open('astOutput.pkl', 'wb') as f:
	#	pickle.dump(program, f)

#    xx1 = countUniqueOps(graph)
#    filename = "./fileGraphDef_LSTM"
#    graph1 = Graph.Graph()
#    with open(filename) as file:
#        graph1.readFromFilePointer(file)
#    xx2 = countUniqueOps(graph1)

if __name__ == "__main__":
	main()
