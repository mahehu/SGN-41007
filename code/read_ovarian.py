# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:27:36 2015

@author: hehu
"""

import numpy as np

if __name__ == "__main__":

    X = [] # Rows of the file go here
    
    # We use Python's with statement. 
    # Then we do not have to worry 
    # about closing it.
    
    with open("ovarian.csv", "r") as fp:
        
        # File is iterable, so we can 
        # read it directly (instead of 
        # using readline).
        
        for line in fp:
            
            # Skip the first line:
            if "Sample_ID" in line:
                continue
            
            # Otherwise, split the line
            # to numbers:
            values = line.split(";")
            
            # Omit the first item 
            # ("S1" or similar):
            values = values[1:]
            
            # Cast each item from
            # string to float:
            values = [float(v) for v in values]

            # Append to X
            X.append(values)            
            
    # Now, X is a list of lists. Cast to 
    # Numpy array:
    X = np.array(X)
    
    print "All data read."
    print "Result size is %s" % (str(X.shape))
    
    