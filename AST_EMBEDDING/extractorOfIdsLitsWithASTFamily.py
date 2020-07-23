#!/usr/bin/env python
import clang.cindex
import sys
from copy import deepcopy
import json
import numpy as np
import os
#Ignioredchild : un noeud de l'ast
#parent : parent de Ignioredchild
#sortie: liste des enfants excepter celui passe en parametre 
def getChildren(parent,IgnioredChild=None):
	children=[]
	for child in parent.get_children():
		"""if(child.kind==clang.cindex.CursorKind.FIELD_DECL ):
			for ch in child.get_children():
				if(IgnioredChild):
					if(ch!=IgnioredChild):
						children.append(ch)
				else :
					children.append(ch)
		else :"""
		if(IgnioredChild):
			if(child!=IgnioredChild):
				children.append(child)
		else:
			children.append(child)
	return children
# ls_node : liste de plusieurs noeuds
# sortie: tous les enfants des noeuds contenu dans ls_node
def getAllChildren(ls_node):
	allChildren=[]
	for node in ls_node:
		childs=getChildren(node)
		for ch in childs:
			allChildren.append(ch)
	return allChildren
#sortie:obtenir la position de child parmis les enfants de parent
def positionIn(parent, child):
	position=getChildren(parent).index(child)
	return position
#afficher un node juste son nom
def nodeToString(node):
	if(node):
		if(node.spelling!=""):
			return node.spelling
		elif (isinstance(node.type,str)):
			return node.type.spelling
		else :
			return str(node.kind)
	else:
		return ""
def lsNodeToString(lsnode,mess):
	print(mess)
	for n in lsnode:
		print(n)
	
#recuperer le context des noeuds partant de la racine
def allParcour(parent,ancestors,all_ID_context):
	for node in parent.get_children():
		positionInParent = positionIn(parent, node)
		if(len(ancestors)>=2):
			grandParent = ancestors[len(ancestors) -2]
			print(len(ancestors),nodeToString(grandParent),nodeToString(parent),nodeToString(node))
			positionInGrandParent = positionIn(grandParent, parent)
			uncles = getChildren(grandParent, parent) 
			cousins = getAllChildren(uncles)
		else :
			grandParent=None
			positionInGrandParent =-1
			uncles = []
			cousins = []
		siblings = getChildren(parent, node)
		
		nephews = getAllChildren(siblings)
		id_context={'token':nodeToString(node),
			  'context':{'parent':nodeToString(parent),
                                     'positionInParent': positionInParent,
                                     'grandParent': nodeToString(grandParent),
                            'positionInGrandParent': positionInGrandParent,
                            'siblings':[nodeToString(n) for n in siblings],
                            'uncles': [nodeToString(n) for n in uncles],
                            'cousins': [nodeToString(n) for n in cousins],
                            'nephews': [nodeToString(n) for n in nephews]
			             }
			   }
		all_ID_context.append(id_context)
		
		
		ancestors.append(node)
		cpAns=[a for a in ancestors]
		cpnode=node	
		allParcour(cpnode,cpAns,all_ID_context)
		ancestors.pop()
if __name__ == '__main__':
	clang_lybrary_path='/usr/lib/llvm-6.0/lib/libclang.so'
	clang.cindex.Config.set_library_file(clang_lybrary_path)
	index = clang.cindex.Index.create()
	
	all_id_context=[]
	with open(sys.argv[1], "r") as fichier:
		path_chaine=fichier.read()
		paths=path_chaine.split('\n')
		for f in paths:
			path=os.path.join(os.getcwd(),f)
			if(os.path.isfile(path)):
				cursor=index.parse(path).cursor
				id_context=[]
				ancestors=[cursor]
				allParcour(cursor,ancestors,all_id_context)
				print(all_id_context)
				
	with open('tokens_with_astcontext.json','w') as fichier:
		json.dump(all_id_context,fichier,sort_keys=True, indent=4)
