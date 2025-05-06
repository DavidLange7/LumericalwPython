# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 18:14:03 2025

@author: dlange
"""

# to start, we will import the necessary environments to make python useful to run
# first we begin with a few "standard" python import 

import numpy as np # a package for handling numerical data
import matplotlib.pyplot as plt # a package for plotting and visualizing data
import sys, os # useful packaged for interacting with the computer
# next we will import the packages required for the lumerical / python API
# sys.path.append(r"C:\Program Files\Lumerical\v242\api\python") # adds the folder containing lumapi to the "path"
sys.path.append(r'C:\Program Files\Lumerical\v242\api\python')
import lumapi # this is the main import required for the lumerical / python API

mode = lumapi.MODE()#filename=file_path,hide=True) # can use filename=file_path argument to open a specific file
#mode.save('modetest.fsp')
#%%
temp2 = mode.addmaterial('Sellmeier') 
Mat_name2 = 'Lithium niobate zcut'
mode.setmaterial(temp2, 'name', Mat_name2)
mode.setmaterial(Mat_name2, 'Anisotropy', 1) #0 or 1 to choose between none and diagonal
mode.setmaterial(Mat_name2, 'color', np.array([100, 50, 50, 255])) #RGB code and lastly the max number of bins, default is 255

sellmeier_no = [2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.6]  # Ordinary (n_o)
sellmeier_ne = [2.9804, 0.02047, 0.5981, 0.06606, 8.9543, 416.08]  # Extraordinary (n_e)

sell_coeff = ['B1', 'C1' ,'B2', 'C2', 'B3', 'C3']
for i in range(len(sell_coeff)):
    mode.setmaterial(Mat_name2, sell_coeff[i], np.array([sellmeier_no[i], sellmeier_no[i], sellmeier_ne[i]]))

temp1 = mode.addmaterial('Sellmeier') 
Mat_name1 = 'Sapphire'
mode.setmaterial(temp1, 'name', Mat_name1)
mode.setmaterial(Mat_name1, 'Anisotropy', 0) #0 or 1 to choose between none and diagonal
mode.setmaterial(Mat_name1, 'color', np.array([0, 100, 100, 255]))
sellmeier_no = [1.43135, 0.00527993, 0.650547, 0.0142383, 5.3414, 325.018]  # Ordinary (n_o)
for i in range(len(sell_coeff)):
    mode.setmaterial(Mat_name1, sell_coeff[i], sellmeier_no[i])
    

#%%
saph_layer = mode.addrect() # this creates a rectangle object
tfln = mode.addrect() 
sin_wg = mode.addrect() 


saph_layer["name"] = "saph_layer" # note, these do not need to be the same, I just like doing it that way
saph_layer["x"] = 0 # there are multiple ways we can set the parameters, either as dictionary calls or object calls
saph_layer["x min"] = -15e-6
saph_layer["x max"] = 15e-6
saph_layer["y min"] = -15e-6
saph_layer["y max"] = 15e-6
saph_layer["z min"] = -4e-6
saph_layer["z max"] = 0e-6
saph_layer["material"] = "Sapphire" # to add the material we need to use the exact string used in the GUI


tfln["name"] = "tfln" 
tfln["x"] = 0
tfln["x min"] = -15e-6
tfln["x max"] = 15e-6
tfln["y min"] = -15e-6
tfln["y max"] = 15e-6
tfln["z min"] = 0
tfln["z max"] = 0.28e-6
tfln["material"] = 'Lithium niobate zcut'

sin_wg["name"] = "sin_wg" 
sin_wg["x"] = 0
sin_wg["x min"] = -15e-6
sin_wg["x max"] = 15e-6
sin_wg["y min"] = -0.3e-6
sin_wg["y max"] = 0.3e-6
sin_wg["z min"] = 0.28e-6
sin_wg["z max"] = 0.98e-6
sin_wg["material"] = 'Si3N4 (Silicon Nitride) - Luke'

#%%
fde_sim = mode.addfde()
meshh = mode.addmesh()

meshh["x"] = 0
meshh["x span"] = 0
meshh["y min"] = -1e-6
meshh["y max"] = 1e-6
meshh["z min"] = 0.1e-6
meshh["z max"] = 1.1e-6

meshh["override x mesh"] = 0
meshh["dy"] = 0.01e-6
meshh["dz"] = 0.01e-6

# defining simulation region
fde_sim["solver type"] = "2D X normal"

fde_sim["x"] = 0
fde_sim["y"] = 0
fde_sim["z"] = 0.5


fde_sim["y min"] = -2e-6
fde_sim["y max"] = 2e-6
fde_sim["z min"] = -0.2e-6
fde_sim["z max"] = 1.2e-6 


# defining boundary conditions. It al
fde_sim["y min bc"] = "PMC"
fde_sim["y max bc"] = "PMC"
fde_sim["z min bc"] = "PMC"
fde_sim["z max bc"] = "PMC"


#%%
mode.run()

def find_modes( wavl, trial_modes):
        mode.switchtolayout()
        mode.setnamed("FDE", "wavelength", wavl)
        mode.setanalysis("number of trial modes", trial_modes)
        return mode.findmodes()
    
def filtered_modes( pol_thres, pol):
    
        mode_ids = [s.split("::")[2] for s in mode.getresult().split('\n')
            if 'mode' in s]
        return [i for i in mode_ids
            if mode.getdata(i, pol+" polarization fraction") > pol_thres]
    
 
def select_mode( mode_id):
        mode.selectmode(mode_id)
        
        
def run_sweep(wavl_center, wavl_span, N_sweep, trial_modes=4, 
                pol_thres=0.96, pol="TE", mode_ind=0):
        # Package simulation data
        wavls = []
        E_fields = []
        H_fields = []
        n_effs = []
        n_grps = []
        # Perform sweep
        wavl_start = wavl_center - wavl_span/2
        wavl_stop = wavl_center + wavl_span/2
        lambdas =  np.linspace(wavl_start, wavl_stop, N_sweep)
        
        for wavl_i in lambdas:
            find_modes(wavl_i, trial_modes)
            mode_id = filtered_modes(pol_thres, pol)[mode_ind]
            select_mode(mode_id)
            E_field = [mode.getdata(mode_id, s)[:,:,0,0] 
                for s in ("Ex","Ey","Ez")]
            n_eff = [mode.getdata(mode_id, "neff")][0]
            H_field = [mode.getdata(mode_id, s)[:,:,0,0] 
                for s in ("Hx","Hy","Hz")]
            n_grp = [mode.getdata(mode_id, "ng")][0]
            E_fields.append(E_field)
            H_fields.append(H_field)
            n_effs.append(n_eff)
            n_grps.append(n_grp)
            wavls.append(wavl_i)
            
        xaxis = mode.getdata("FDE::data::material", "x")
        yaxis = mode.getdata("FDE::data::material", "y")
        index = mode.getdata("FDE::data::material", "index_y")
        
        c = 2.99792e-7
        lambdas =  np.linspace(wavl_start, wavl_stop, N_sweep) * 10**(9)
        omegas = 2*np.pi*c* lambdas
        n_effs = np.reshape(n_effs, len(n_effs))
        
        dneff = np.gradient(n_effs, lambdas) 
        d2neff = np.gradient(dneff, lambdas)


                
        D = - lambdas/c * d2neff
        
        plt.plot(lambdas, D)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Dispersion D (ps/nm/km)")
        plt.title("Dispersion vs Wavelength")
        plt.grid(True)
        plt.show()
        
        
            
        return [xaxis, yaxis, index, wavls, np.array(E_fields), 
            np.array(H_fields), np.array(n_grps), np.array(n_effs)]
    



#%%
mode.close()
