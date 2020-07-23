# implementation_bug_detector
Overview
----------
*AST_EMBEDDING : contient l'implementation de la AST-EMBEDDING pour le language c/c++

*cnn-text-classification: contient l'implementation du CNN pour la classification binaire(texte positif ou negatif) de texte 

requiments
------------
*AST_EMBEDDING

   -compilateur clang : module qui nous permet de manipuler l'ast
       sudo apt install clang
       sudo apt install libclang-dev
       
   -module python
      - clang
      - regex
      - subprocess32a
      -json
      
*CNN-TEST-CLASSIFICATION

   -numpy
   
   -tensorflow version=1
   
-Learning AST_EMBEDDING :
--------------------------
  Executer les etapes suivantes depuis le dossier AST_EMBEDDING
  
  step 1:
  
     python extractorOfIdsLitsWithASTFamily.py filepath 
     
  step 2:
  
     python TokenWithASTContextToNumbers.py tokens_with_astcontext.json   
     
  step 3:
  
     python ASTEmbeddingLearner.py main_token_to_number_*.json  encoded_tokens_with_context_*.npy
     
     main_token_to_number_*.json  encoded_tokens_with_context_*.npy sont des fichiers  produits a l'etape 2 
     
  on obtient le fichier: token_to_vector_1595511405913.json qui contient la representation vectoriel de chaque mot
  
-Entrainer et Evaluer  CNN :
--------------------------------
Executer les etapes suvantes  depuis le dossier cnn-text-classification :

  step 1 : entrainer le CNN
  
     python train.py
     
  step 2 : evaluer le CNN
  
     python eval.py
