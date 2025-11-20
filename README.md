The stage2.py, stage3.py, and stage4.py are the training files of the different degradation stage. 

The compressed file of checkpoint needs to be decompressed before it can be used.
If a checkpoint is required, the last word representing the stage needs to be removed. 
For example, you need to change the "informer_NASA_ftMS_sl5_ll5_pl1_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_False_stage3" to "informer_NASA_ftMS_sl5_ll5_pl1_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_False"

The results of running the code are stored in the results folder, and their automatic naming is based on the hyperparameters in the algorithm. In the existing results, for the convenience of comparison, we added a word to the folder corresponding to the degradation stage.
Therefore, the predicted RUL results can be seen in the results file. 
In the results file, you can see three sub files corresponding to RUL prediction results for different degradation stages, with the last word of the file name corresponding to the degradation stage. 
In addition, using the read.exe file can draw a comparison chart between predicted values and actual values.
