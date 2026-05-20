Contents:

1. caros-beach-scribble-mask.mat 

contains the scribble mask image (where a pixel has value 1 
when it lies on the foreground scribble, 2 when it lies on 
the background scribble, and 0 otherwise).


2. carlos-beach-sp.mat contains a few variables of interest:

   labels -- which is a superpixel map, where each pixel holds 
             the index of the superpixel it belongs to.

   regsize -- number of pixels in each superpixel.

   modes -- average Luv vectors for each superpixel.


This superpixel data was computed using code available from:
http://www.caip.rutgers.edu/riul/research/code.html

 
