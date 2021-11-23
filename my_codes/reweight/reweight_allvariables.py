import numpy as np
import matplotlib.pyplot as plt

def get_reweighted_allbkg_features(sets_of_npzs, weights, mz400_template, mz400_xsec, n_target_events=30000, n_template = 2000):
    """
    Gets a combined, weighted combination of events
    """
    weights /= np.sum(weights) # normalize
    mz400_xsec /= np.sum(mz400_xsec)
    print(f'weights: {weights}')
    n_events_per_set = (weights * n_target_events).astype(np.int32)
    nsig_events_per_set = (mz400_xsec *n_template).astype(np.int32)
    print(f'n_events per set: {n_events_per_set}')
    
    # this section to produce the signal pt histogram 
    #I used the mz'=400 as the template histogram    
    for ns_events, nsig_pzs in zip(nsig_events_per_set, mz400_template):
        n_events_todo = ns_events
        mz400_comb = []        
        for npz in nsig_pzs:
            X = np.load(npz)['X'][:,:12]
            n_events_this_npz = X.shape[0]
            if n_events_this_npz > n_events_todo:
                X = X[:n_events_todo]
            mz400_comb.append(X)
    mz400_final = np.vstack(mz400_comb)
    pt_sig = mz400_final[:,11]

    
    #need to debug
    #debug1: the error about mismaching between w_pt and X_combined 
    #comp_X = 0 # debug1
    #comp_w = 0 # debug1
    
    X_combined = []
    W_pt = []
    for n_events, npzs in zip(n_events_per_set, sets_of_npzs):
        n_events_todo = n_events
        for npz in npzs:
            X = np.load(npz)['X'][:,:12]
            #print(f'x shape: {X.shape}')
            n_events_this_npz = X.shape[0]
            if n_events_this_npz > n_events_todo:
                X = X[:n_events_todo]
            X_combined.append(X)
            
            #print(f'the npz file is: {npz}') #debug1 
            #comp_X += X.shape[0] #debug1
            
            
            pt = []
            pt_qcd = np.load(npz)['X'][:,11] 
            fig = plt.figure(figsize=(6,6))
            ax1 = fig.gca()
            pt.append(pt_qcd)
            pt.append(pt_sig)
            ns, bins, patches = ax1.hist(pt, bins=50, alpha=0.2, range=[0,1000], label=['qcd','sig'])
            ax1.legend()
            
            w_pt = np.zeros((n_events_this_npz,1))
            #print(f'wpt initialized: {w_pt.shape}') #debug1           
                        
            for i in range(len(pt_qcd)):
              for j in range(len(ns[1])):
               if bins[j]<pt_qcd[i] and bins[j+1]>pt_qcd[i] and ns[1,j] > 0:
                w_pt[i] = ((ns[1][j]/ns[0][j]))
               
            if n_events_this_npz > n_events_todo:
                w_pt = w_pt[:n_events_todo]
            W_pt.append(w_pt)          
            
            #comp_w += w_pt.shape[0] #debug1 
            #print(f'shape of X is: {X.shape[0]} and for w_pt is: {w_pt.shape[0]}') #debug1 
            plt.close() #to close the display of the figures --> avoid memory intensive job
            n_events_todo -= X.shape[0]
            if n_events_todo == 0: break        
        #else:
        #    print(f'Warning: reached end of set of files with n_events_todo={n_events_todo}')
            
    print(f'X_combine length: {len(X_combined)}') #sanity check
    print(f'wpt length: {len(W_pt)}') #sanity check
    

    #print(f'multiplication of x_combine and w_pt {X_combined*w_pt}')
    #print(f'x_combined: {X_combined}')
    #print(f'w_pt: {w_pt}')
    #print(f'completed shape for w_pt: {comp_w}, and for X is: {comp_X}') #debug1
    

    
    X_final = np.vstack(X_combined)
    wpt_final = np.vstack(W_pt)
    
    #this is completed wrong --> the weight should be applied into the weight and not the values of the parameters
    #X_3 = X_final*wpt_final
    #X_3 = np.multiply(X_final, wpt_final)
    #X_Finale = np.column_stack((X_final,wpt_final)) #to add w_pt at the end of the array
    
    
    print(f'**********************************************')
    
    print(f'X_final shape: {X_final.shape}')     #size of stacked X_combined array
    print(f'wpt_final shape: {wpt_final.shape}') #size of stacked w_pt array
    #print(f'X_Finale shape: {X_Finale.shape}')
    #print(f'X_multiplication shape: {X_3.shape}')
    print(f'**********************************************')
    
    #assert len(X_Finale.shape) == 2
    #print(f'X_Finale: {X_Finale[1,:]}')   #check w_pt is the last element in the array
    print(f'wpt_final: {wpt_final[1,:]}') #make sure it's computed correctly
    print(f'      ')
    print(f'X_final: {X_final[1,:]}')
    print(f'      ')
    #print(f'_multiplication: {X_3[1,:]}')
    #assert X_Finale.shape[1] == 13
    #print(f'X_3 nine elements: {X_3[:,:9]}')
    #return X_3[:,:9]

    #weights /= np.sum(weights) # normalize
    #print(f'weights: {weights}')
    #print(f'**********************************************')

    
    #solved the multiplication error
    #the factors shouldn't be applied to the value of the elements but to the number of events
    #todo: 
      #how to interpret the pt_weight: w_pt = pt_sig/pt_bkg and then X_final*w_pt or the other way
      #clean up the code and send it to github
    
    
    X_COM = []
    for n_events, npzs in zip(n_events_per_set, sets_of_npzs):
        n_events_todo = n_events*wpt_final
        for npz in npzs:
            X = np.load(npz)['X'][:,:]
            n_events_this_npz = X.shape[0]*wpt_final
            if n_events_this_npz.any() > n_events_todo.any():
                X = X[:n_events_todo]
            X_COM.append(X)
            if n_events_todo.any() == 0: break        
        #else:
        #    print(f'Warning: reached end of set of files with n_events_todo={n_events_todo}')
    X_FIN = np.vstack(X_COM)
    assert len(X_FIN.shape) == 2
    #assert X_FIN.shape[1] == 9
    print(f'shape of X_FIN: {X_FIN.shape}')
    
    print(f'X_final: {X_final[1,:10]}')
    print(f'X_FIN value: {X_FIN[1,:10]}')
    return X_FIN
