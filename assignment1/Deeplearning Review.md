## Deeplearning Review       

### Supervised learning 

* these adjustable parameters,often called weights
* gradient vector
* objective function 
* stochastic gradient desent（SGD）
* SGD process is repeated for many small sets of examples       from the training set until the average of the objective      function stops decreasing    
* optimization   
* generalization ability - its ability to produce sensible      answers on new inputs that it has never seen during training
* threshold
* shallow classifier 
* feature extractor
* selectivity-invariance-dilemma    
* generic non-linear features,as with kernel method   
* But this can all be avoided if good features can be learned   automatically using a general-purpose learning procedure,     This is the key advantage of deep learning 
* deep learning architecture is multilayer stack           


### BackPropagation to train multilayer architectures      

* the aim of researchers has been to replace hand-engineered features with trainable multilayer networks,but despite its simplicity,the solution was not widely understood until the mid 1980s. As it turns out,multilayer architectures can be trained by simple stochastic gradient descent.        
* chain rule for derivatives       
* once these gradients(对目标函数) have been computed, it is straightforward to compute the gradients(目标函数对每个权值) with respect to the weights of each moudule
* feedforward neural network architectures       
* non-linear function   
* rectified linear unit(ReLU)     
* units that are not in the input or output layer are conventionally called hidden units        
* local minima   
* saddle points where the gradient is zero       
* CIFAR introduced unsupervised learning procedures that could   
* overfitting    
* full connectivity between adjacent layers      
* convolutional neural network(ConvNet)     


### Convolutional neural networks      

* local connections(局部连接)      
* shared weights(权值共享)          
* pooling(池化)    
* the use of many layers     
* units in a convolutional layer are organized in feature maps    
* filter bank   
* local motif(局部特征)     
* the role of the convolutional layer is to detect local conjunctions of features from the previous layer    
* the role of the pooling layer is to merge semantically similar features into one      
* hierarchies(层级)    
* neocognitron(神经认知)     
* time-delay neural net    


### Image understanding with deep convolutional networks      

* segmentation and recognition       
* ImageNet competition        
* a new regularization technique called dropout     
* field-programmable gate array(FPGA)     
* ConvNet chips      
* real-time vision applications(实时视觉应用)    


### Distributed representations and language processing     

* exponential int the depth   
* micro-rules      
* distributed representations     
    * their elements are not mutually exclusive     
    * their many configurations correspond to the variations seen in the observed data     
* logic-inspired paradigm     
* N-grams     


### Recurrent neural networks     

* state vector    
* RNNs are very powerful dynamic system(动态系统)    
* thought vector    
* learn long-term dependencies        
* long short-term-memory(LSTM)     
* augment RNNs with a memory module   
* Neural Turing Machine      
* memory networks    


### The feature of deep learning     

* Human and animal learning is largely unsupervised:  we discover the structure of the world bu observing it, not by being told the name of every object      
* combine ConvNets with RNNs    
* we expect systems that use RNNs to understand sentences or whole documents     
* artificial intelligence(人工智能)    
* combine representation learning with complex reasoning(结合表示学习和复杂推理)     
