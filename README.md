# noahroselli-schrodingerproject
A brief project constructed in Python based in object oriented programming 
that shows the time evolution of a wave function in varying potentials.

For the user of this Schrodinger 2-D numerical solution: 

To use this effectively, note that the only changes that you should make to the files
are contained within the 'equations.py' file. 
If you want to change which potential the solution is running, the method of plotting, 
the initial condition, etc., all of this is contained within 'equations.py'.

'timesteppers.py' contains the necessary timestepping methods to allow the numerical 
solution of the differential equation to happen. 

'schrodinger.py' contains the classes that are called in 'equations.py', with each class
corresponding to a different potential. 

'animate.py' is a file that allows the animation to occur. Note: you must exit the Python
plot in order to stop the animation from continuing. 

'farray.py' is an external file that allows certain functions to run. Do not modify. 

If you have any questions while using it, email:
noahroselli2027@u.northwestern.edu
