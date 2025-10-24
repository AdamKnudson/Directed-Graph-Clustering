# DiGraph Clustering

## Goal:
Use Algorithm 1 from "Hermitian matrices for clustering directed graphs: insights and applications" 
to create an iterative algorithm that will find a better clustering for a given digraph, given we know the true meta graph.

## Work in progress
These functions are not doing as well as we'd like yet. This is ongoing work started from a 285J class project.

Also a note: need to correct a small bug in Algorithm 1. We are currently taking the k eigenvectors corresponding to the 
k largest eigenvalues, but we should be taking the k largest eigenvalues in magnitude. 
Even with this error, the results of the algorithm seemed to match quite well with the results found in the original paper
so it is unclear if this could significantly improve these Algorithms, however, it should still get fixed.
