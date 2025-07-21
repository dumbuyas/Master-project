

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.fem import assemble_scalar
from dolfinx.fem.petsc import assemble_vector
import pyvista 

import utils
from dolfinx.io import XDMFFile
import dolfinx.fem as fem
import dolfinx.plot as plot
import numpy as np
import dolfinx
import ufl
import time 
comm = MPI.COMM_WORLD
pyvista.OFF_SCREEN = True
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numProc = comm.Get_size()
import psutil, os

start_time = time.time()




# define the optimization problem 
class TAOProblem:

    def __init__(self, J_form: ufl.form.Form, gradJ_form: ufl.form.Form, rho: dolfinx.fem.Function, uh:dolfinx.fem.Function, PDEproblem,options, Compliance_ufl,Perimeter_ufl, Penalty_ufl, output,mesh):


        self.J_form = J_form  #  The objective function 
        self.gradJ_form = gradJ_form  # The gradient of the objective function
        self.rho = rho    #The design variable 
        self.PDEproblem=PDEproblem  # The PDE constraint 
        self.uh=uh                   # the solution of the PDE constraint 
        self.J_final = None           #store the final computed objective function value   
        self.options=options       # command line options 
        self.Compliance_ufl=Compliance_ufl # the compliance term
        self.Perimeter_ufl=Perimeter_ufl    # The Perimeter term
        self.Penalty_ufl= Penalty_ufl        #  The Penalty term
        self.prefix=output
        self.mesh=mesh
                                      

          

        self.plotter = pyvista.Plotter(off_screen=True)  # Initialize a persistent PyVista Plotter
        utils.plot_density(self.rho, f'{self.prefix}-initial.png') #Initial design

    def ObjectiveFunction(self, tao, x:PETSc.Vec):
        #Evaluates the objective function for the TAO solver
        # Update the design variable rho
        # Solve the PDE constraint
        #Computes and returns the J_global

        with self.rho.vector.localForm() as loc:
            loc[:] = x.array                      
        self.rho.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.uh = self.PDEproblem.solve()
        self.J_final= dolfinx.fem.assemble_scalar(self.J_form)    
        J_global = MPI.COMM_WORLD.allreduce(self.J_final, op=MPI.SUM)  
        return (J_global)
        


     #Compute the gradient of the objective function
    def GradObjectiveFunction(self, tao: PETSc.TAO, x:PETSc.Vec, G):
        #Compute the gradient of the objective function 
        #update the design variable rho
        # Solve the PDE constraint 

       
        with self.rho.vector.localForm() as loc:
            loc[:] = x.array  # Update rho from PETSc TAO


        self.rho.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

         # solve the PDE
        self.uh = self.PDEproblem.solve()

        with G.localForm() as J_form_local: 
             J_form_local.set(0.0)  # Reset previous values
        G.set(0.0)
        

        assemble_vector(G, self.gradJ_form)  # Assemble directly into G
        G.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # Synchronize across MPI processes 
        

        return G
    
     #define tao monitor
    def Monitor(self,tao):
       
        dx = ufl.Measure("dx", domain=self.rho.function_space.mesh)

        iter_num = tao.getIterationNumber()
        f_val = tao.getObjectiveValue()
      

        volume = fem.assemble_scalar(fem.form(1.0 * dx))   #the volume 
        target_volume= 0.3*volume    # define the target volume

        

      
        G, *_ = tao.getGradient()  #extract the gradient vector 
        grad_norm = (G.norm(PETSc.NormType.NORM_2))

        
        save_interval=self.options.get("saveinterval", 10)
        if iter_num %save_interval==0:
            if numProc == 1:
                utils.plot_density(self.rho, f'{self.prefix}-{iter_num:02d}.png')
                self.plotter.close() 
               # self.f.write_function(self.rho, iter_num * self.options['mu'])
            self.rho_min = comm.allreduce(np.min(self.rho.vector.getArray()), op=MPI.MIN)
            self.rho_max = comm.allreduce(np.max(self.rho.vector.getArray()), op=MPI.MAX)
            self.uh_max = comm.allreduce(np.max(self.uh.x.array), op=MPI.MAX)
            self.uh_min = comm.allreduce(np.min(self.uh.x.array), op=MPI.MIN)

            # Compute compliance, perimeter, penalty
            self.Compliance = comm.allreduce(fem.assemble_scalar(fem.form(self.Compliance_ufl)), op=MPI.SUM)
            self.Perimeter = comm.allreduce(fem.assemble_scalar(fem.form(self.Perimeter_ufl)), op=MPI.SUM)
            self.Penalty = comm.allreduce(fem.assemble_scalar(fem.form(self.Penalty_ufl)), op=MPI.SUM)
            #Obj = Compliance + Perimeter + Penalty
            self.current_volume=fem.assemble_scalar(fem.form(self.rho*dx))
            
        
           
            self.rho.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
              
        if rank == 0:
            print(f"[TAO Monitor] Iter {iter_num}:, Obj: {f_val:.4e},GradNorm: {grad_norm:.6e} ",
            f"rho_min={self.rho_min:.2e}, rho_max={self.rho_max:.2e}, uh_min={self.uh_min:.4e}, uh_max={self.uh_max:.4e},"
            f"Compliance={self.Compliance:.4e}, Perimeter={self.Perimeter:.4e}, Penalty={self.Penalty:.4e}, target_volume={target_volume:.4e}, current_volume={self.current_volume:.4e}, ")
            end_time = time.time()

            convergence_time = end_time - start_time
            print(f"Convergence time: {convergence_time:.2f} seconds")
           # print(f"Peak Memory: {peak_memory:.2f} MB")
       
            
    



        