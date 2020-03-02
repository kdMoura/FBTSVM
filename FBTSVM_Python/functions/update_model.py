## APPROXIMATE KERNEL FUNCTION
## Currently there are only the sklearn functions available

import pdb
import numpy as np
import pandas as pd
import sys
sys.path.append("/media/alexandre/57268F1949DB0319/MATLAB/FBTSVM/FBTSVM_Python/functions/")
from approx_k import approx_kernel
from fuzzy import fuzzy_membership
from calc import calc_train
from aux_functions import data_structure
import math



def inc_model(parameters,traindata,trainlabel):
    print("D")


def update_model(parameters,data_x,data_y,batch_size,model):
    #Implement to use the appox kernel function

    #The Batch size is a int that represent the number of instances

    data_size=len(data_y)
    integertest=batch_size==math.floor(batch_size)

    if integertest==True:
        bats=batch_size
    else:
        bats=batch_size*data_size
        bats=math.floor(bats)

    p=0

    for i in range(data_size):
        traindata1=data_x[p:p+bats,:]
        trainlabel1=data_y[p:p+bats]
        pdb.set_trace()
        #TODO WORK ON THE INC_MODEL FUNCTION
        fbtsvm_structu,datau,labelu=inc_model(parameters,traindata1,t trainlabel1,model)
        print("update model")


    classes=np.unique(data_y)
    #get the number of classes
    num_classes=len(classes)
    fbtsvm_struct=[]

    #Select a class

    for currentclass in classes:
        #print("loop over two classes only for the DAG algorithm")
        #Xp - all training data from one class
        Xpi=np.where(data_y==currentclass) # indexes from data which are equal to the respective class
        Xp=data_x[Xpi] # The data from the indexes above
        lp=len(Xpi[0]) # the number of instances
        Lp=np.ones(lp) #array of ones
        #pdb.set_trace()
        otherclasses=np.delete(classes,np.where(classes==currentclass))

        #pdb.set_trace()

        if len(fbtsvm_struct)!=0:
            for mod in fbtsvm_struct:
                cl_pos=mod.currentclass
                #pdb.set_trace()
                if mod.ocl==currentclass & currentclass>mod.currentclass:
                    otherclasses=np.delete(otherclasses,np.where(otherclasses==mod.currentclass))
                    #pdb.set_trace()


        #check if the structure already exists
        #print(currentclass)
        #pdb.set_trace()
        if len(otherclasses)>0:

            for ocl in otherclasses:
                Lpi=np.where(data_y==ocl) # indexes from data which are equal another class (negative from currentclass)
                Xn=data_x[Lpi] # The data from the indexes above
                ln=len(Lpi[0]) # the number of instances
                Ln=-1*np.ones(ln) #array of ones
                #pdb.set_trace()
                X=np.concatenate((Xp,Xn))
                L=np.concatenate((Lp,Ln))
                #The fuzzy function can be improved
                sp,sn,NXpv,NXnv=fuzzy_membership(Xp,Xn,parameters);


                #XP_one and XN_one variables
                #pdb.set_trace()
                XP_one=np.append(Xp, np.ones((len(Xp), 1)), axis=1)
                XN_one=np.append(Xn, np.ones((len(Xn),1)),axis=1)
                CCp=parameters.iloc[0].loc['CC']*sn
                CCn=parameters.iloc[0].loc['CC2']*sp
                #pdb.set_trace()

                #Implement here the main function (calc)
                CR2=parameters.iloc[0].loc['CR2'] #for the positives
                CR=parameters.iloc[0].loc['CR'] #for the negatives
                alpha,vp,iter,pgp=calc_train(XN_one,XP_one,CR2,CCn,parameters)
                beta,vn,iter2,pgn=calc_train(XP_one,XN_one,CR,CCp,parameters)
                vn=-vn
                new_structure=data_structure(sp,sn,alpha,beta,vp,vn,NXpv,NXnv,pgp,pgn,currentclass,ocl,Xpi,Lpi)
                fbtsvm_struct.append(new_structure)
            #print("Calculated")

            #pandas dataframe in currentclass and ocl
            #todo save the structure (ftsvm_struct)




    return fbtsvm_struct
