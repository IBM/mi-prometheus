Configuration file
-------------------


Following, we need to create the corresponding configuration file (named mnist_lenet5.yaml) to execute the Trainer (Fig. 4). 
First, we define the Model section, which is straight forward, as the Model does not have any additional parameters. 
In the training section, we define the problem (name and required parameters), optimizer and terminal conditions. 
We select the Adam optimizer [KB14] and set an upper limit of 20 epochs. 
The section used by the Tester requires only the Problem’s definition and indicates that we use the test set.
