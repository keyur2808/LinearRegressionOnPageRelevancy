**************************************************************************************************************************************************************************

                                         INTRODUCTION TO MACHINE LEARNING CSE 574 - PROJECT 1 
                                                LINEAR REGRESSION WITH BASIS FUNCTIONS


**************************************************************************************************************************************************************************

1.The main file is 'project1.m' handles the entire program execution.Its a self contained file that calls other functions files associated with it.
2.It reads the project1_data.mat file to populate the dataset.
3.There are functions files corresponding to each of the training and testing methods.
   
    train_cfs implements training for maximum likely hood
    It takes M - model complexity , lambda_cfs the regularization coefficient and input and output vector as input parameters.
	
	train_gd implements training for Stochastic gradient descent
	It takes M - model complexity ,coefficient and input and output vector as input parameters.
	
	test_cfs takes test data ,weight matrix,parameters for design matrix and returns the rms_cfs error
	test_gd takes test data ,weight matrix,parameters for design matrix and returns the rms_gd error
	
4.It prints the user name,number and results of testing and the parameters used.	
	