# implementation_bug_detector
Overview
----------
*AST_EMBEDDING : contient l'implementation de la AST-EMBEDDING pour le language c/c++

*cnn-text-classification: contient l'implementation du CNN pour la classification binaire(texte positif ou negatif) de texte

requiments
------------
*AST_EMBEDDING

   -compilateur clang et ses dependances  : module qui nous permet de manipuler l'ast
       sudo apt install clang
       sudo apt install libclang-dev
       
   -module python
      - clang
      - regex
      - subprocess32a
      -json
      
*cnn-text-classification

   -numpy
   -tensorflow version=1
   
-Learning AST_EMBEDDING :
--------------------------
  se mettre dans  dossier AST_EMBEDDING
  
  step 1: 
     python extractorOfIdsLitsWithASTFamily.py filepath 
     
  step 2:
     python TokenWithASTContextToNumbers.py tokens_with_astcontext.json   
     
  step 3:
     python ASTEmbeddingLearner.py main_token_to_number_*.json  encoded_tokens_with_context_*.npy
     ces fichiers sont ceux produit a l'etape precedente 
     
  on obtient le fichier: token_to_vector_1595511405913.json qui contient la representation vectoriel de chaque mot
  
-Entrainer et Evaluer  CNN :
--------------------------------
se mettre dans le dossier cnn-text-classification :

  step 1 : entrainer le CNN
     python train.py
     
  step 2 : evaluer le CNN
     python eval.py
