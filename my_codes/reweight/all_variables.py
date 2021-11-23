import numpy as np


def get_allbkg_features(sets_of_npzs, weights, n_target_events=30000):
    """
    Gets a combined, weighted combination of events
    """
    weights /= np.sum(weights) # normalize
    print(f'weights: {weights}')
    n_events_per_set = (weights * n_target_events).astype(np.int32)

    print(f'n_events per set: {n_events_per_set}')
    X_combined = []
    for n_events, npzs in zip(n_events_per_set, sets_of_npzs):
        print(f'n_events: {n_events}')
        n_events_todo = n_events
        for npz in npzs:
            X = np.load(npz)['X'][:,:]
            n_events_this_npz = X.shape[0]
            if n_events_this_npz > n_events_todo:
                X = X[:n_events_todo]
                
            X_combined.append(X)
            n_events_todo -= X.shape[0]
            if n_events_todo == 0: break
        else:
            print(f'Warning: reached end of set of files with n_events_todo={n_events_todo}')
    X_final = np.vstack(X_combined)
    
    assert len(X_final.shape) == 2
    #assert X_final.shape[1] == 9
    return X_final
