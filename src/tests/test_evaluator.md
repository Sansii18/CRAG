This file is quality check for Step 3 

It mainly checks two things : 
1. Evaluator -> Can it generate a valid confidence score?
2. Router -> Can it choose the correct action based on that confidence score?

We are not testing Qdrant , Embeddings or retrieval here. 

Flow of "test_evaluator_scoring()" -> 
![alt text](2026-06-13_23-50-50.png)

Flow of "test_router_routing()" ->
![alt text](image.png)

![alt text](image-1.png)