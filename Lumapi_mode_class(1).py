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
import scipy
from scipy.interpolate import griddata
import time
from scipy.constants import c
import time
from scipy.interpolate import UnivariateSpline
import csv
# next we will import the packages required for the lumerical / python API
# sys.path.append(r"C:\Program Files\Lumerical\v242\api\python") # adds the folder containing lumapi to the "path"
sys.path.append(r'C:\Program Files\Lumerical\v242\api\python')
import lumapi # this is the main import required for the lumerical / python API

sys.path.append(os.path.abspath("C:/Users/dlange/Desktop/Codes"))
#import NLS_solver as ns
#import gnlse

eps0 = 8.854*10**(-12)
mu0 = 4*np.pi*10**(-7)
#%%


'''
#useful commands to know:
    
In case you want to delete structures:
self.mode.select('saph_layer')
self.mode.delete()
self.mode.select('sin_wg')
self.mode.delete()
self.mode.select('tfln')
self.mode.delete()
here you can also give:
    ("solver type","2D Z normal"),
             ("wavelength",wavelength),
             ("number of trial modes",num_modes),
             ("bent waveguide",bend_waveguide),
             ("bend radius",bend_radius),
             ("bend orientation",bend_orientation))),  
    )

use self.mode.setnamed("name", "property", value)
Find results:
    e.g.
    self.mode.getdata("FDE::data::mode1","neff")
    self.mode.getdata("FDE::data::mode1","TE polarization fraction")
    self.mode.getelectric("FDE::data::mode1")
    self.mode.getmagnetic("FDE::data::mode1")
    
    # Turn on redraw feature to update simulation layout
       self.mode.redrawon()
        self.mode.save(filename)
        
    interact with FDE:
        self.self.mode.selectmode(mode_id)
        
        def bent_waveguide_setup(bend_radius, orientation_angle, 
            x_bend=None, y_bend=None, z_bend=None):
            self.mode.setanalysis("bent waveguide", True) 
            self.mode.setanalysis("bend radius", bend_radius)
            self.mode.setanalysis("bend orientation", orientation_angle)
            self.mode.setanalysis("bend location", 1)
            if (x_bend or y_bend or z_bend) != None:
                self.mode.setanalysis("bend location", 2)
                self.mode.setanalysis("bend location x", x_bend)
                self.mode.setanalysis("bend location y", y_bend)
                self.mode.setanalysis("bend location z", z_bend)

'''


class lumerical_sim:
    def __init__(self):
        
        self.mode = lumapi.MODE()#filename=file_path,hide=True) # can use filename=file_path argument to open a specific file

    def addmaterial(self):
        temp2 = self.mode.addmaterial('Sellmeier') 
        Mat_name2 = 'Lithium niobate zcut'
        self.mode.setmaterial(temp2, 'name', Mat_name2)
        self.mode.setmaterial(Mat_name2, 'Anisotropy', 1) #0 or 1 to choose between none and diagonal
        self.mode.setmaterial(Mat_name2, 'color', np.array([100, 50, 50, 255])) #RGB code and lastly the max number of bins, default is 255

        sellmeier_no = [2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.6]  # Ordinary (n_o)
        sellmeier_ne = [2.9804, 0.02047, 0.5981, 0.06606, 8.9543, 416.08]  # Extraordinary (n_e)
    
        sell_coeff = ['B1', 'C1' ,'B2', 'C2', 'B3', 'C3']
        for i in range(len(sell_coeff)):
            self.mode.setmaterial(Mat_name2, sell_coeff[i], np.array([sellmeier_no[i], sellmeier_no[i], sellmeier_ne[i]]))
        
        temp1 = self.mode.addmaterial('Sellmeier')
        Mat_name1 = 'Sapphire'
        self.mode.setmaterial(temp1, 'name', Mat_name1)
        self.mode.setmaterial(Mat_name1, 'Anisotropy', 0) #0 or 1 to choose between none and diagonal
        self.mode.setmaterial(Mat_name1, 'color', np.array([0, 100, 100, 255]))
        sellmeier_no = [1.43135, 0.00527993, 0.650547, 0.0142383, 5.3414, 325.018]  # Ordinary (n_o)
        for i in range(len(sell_coeff)):
            self.mode.setmaterial(Mat_name1, sell_coeff[i], sellmeier_no[i])
            
        temp3 = self.mode.addmaterial('Sellmeier') 
        Mat_name3 = 'Al2O3 amorphous'
        self.mode.setmaterial(temp3, 'name', Mat_name3)
        self.mode.setmaterial(Mat_name1, 'Anisotropy', 0) #0 or 1 to choose between none and diagonal
        self.mode.setmaterial(Mat_name1, 'color', np.array([50, 10, 100, 100]))
        sellmeier_no = [1.260, 0.156**2, 0.484, 0.093**2, 59.186, 32.3**2]  # Ordinary (n_o)
        for i in range(len(sell_coeff)):
            self.mode.setmaterial(Mat_name3, sell_coeff[i], sellmeier_no[i])
            
        temp4 = self.mode.addmaterial('(n,k) Material')
        Mat_name4 = 'aluladoua'
        self.mode.setmaterial(temp4, 'name', Mat_name4)
        self.mode.setmaterial(Mat_name4, 'Anisotropy', 0)
        self.mode.setmaterial(Mat_name4, 'color', np.array([10, 200, 100, 10]))
        self.mode.setmaterial(Mat_name4, 'Refractive Index', 1.6)
        self.mode.setmaterial(Mat_name4, 'Imaginary Refractive Index', 0)
        
        temp5 = self.mode.addmaterial('(n,k) Material')
        Mat_name5 = 'SiN La doua'
        self.mode.setmaterial(temp5, 'name', Mat_name5)
        self.mode.setmaterial(Mat_name5, 'Anisotropy', 0)
        self.mode.setmaterial(Mat_name5, 'color', np.array([10, 100, 200, 10]))
        self.mode.setmaterial(Mat_name5, 'Refractive Index', 2.2)
        self.mode.setmaterial(Mat_name5, 'Imaginary Refractive Index', 0)

    
    def updatestructures(self, name, ymm, zmm, material):
        self.mode.select(name)
        self.mode.delete()
        
        newlayer = self.mode.addrect()
        
        newlayer["name"] = name
        newlayer["x"] = 0
        newlayer["x min"] = -15e-6
        newlayer["x max"] = 15e-6
        newlayer["y min"] = ymm[0]
        newlayer["y max"] = ymm[1]
        newlayer["z min"] = zmm[0]
        newlayer["z max"] = zmm[1]
        newlayer["material"] = material

    def addstructures(self):
        saph_layer = self.mode.addrect() # this creates a rectangle object
        tfln = self.mode.addrect() 
        sin_wg = self.mode.addrect() 


        saph_layer["name"] = "saph_layer" # note, these do not need to be the same, I just like doing it that way
        saph_layer["x"] = 0 # there are multiple ways we can set the parameters, either as dictionary calls or object calls
        saph_layer["x min"] = -15e-6
        saph_layer["x max"] = 15e-6
        saph_layer["y min"] = -50e-6
        saph_layer["y max"] = 50e-6
        saph_layer["z min"] = -10e-6
        saph_layer["z max"] = 0e-6
        saph_layer["material"] = "Sapphire" # to add the material we need to use the exact string used in the GUI


        tfln["name"] = "tfln" 
        tfln["x"] = 0
        tfln["x min"] = -15e-6
        tfln["x max"] = 15e-6
        tfln["y min"] = -50e-6
        tfln["y max"] = 50e-6
        tfln["z min"] = 0
        tfln["z max"] = 0.28e-6 #or 0.28
        tfln["material"] = 'Lithium niobate zcut'

        sin_wg["name"] = "sin_wg" 
        sin_wg["x"] = 0
        sin_wg["x min"] = -15e-6
        sin_wg["x max"] = 15e-6
        sin_wg["y min"] = -0.5e-6
        sin_wg["y max"] = 0.5e-6
        sin_wg["z min"] = 0.3e-6
        sin_wg["z max"] = 1.1e-6
        sin_wg["material"] = 'Si3N4 (Silicon Nitride) - Luke'
        
    def addmesh(self, ymm, name, zmm = [-0.5e-6, 2e-6]):
        
        self.mode.switchtolayout()
        
        self.mode.select(name)
        self.mode.delete()
            
        globals()[name] = self.mode.addmesh()
        
        globals()[name]["name"] = name
        globals()[name]["x"] = 0
        globals()[name]["x span"] = 0
        globals()[name]["y min"] = ymm[0]
        globals()[name]["y max"] = ymm[1]
        globals()[name]["z min"] = zmm[0]
        globals()[name]["z max"] = zmm[1]

        globals()[name]["override x mesh"] = 0
        globals()[name]["dy"] = 0.01e-6
        globals()[name]["dz"] = 0.01e-6
        
    def addsimulation(self, ymm = [-30e-6, 30e-6]):
        
        self.mode.switchtolayout()
        self.mode.select('FDE')
        self.mode.delete()
        
        fde_sim = self.mode.addfde()
        

        # defining simulation region
        fde_sim["solver type"] = "2D X normal"
        
        fde_sim["x"] = 0
        fde_sim["y"] = 0
        fde_sim["z"] = 0.5

        fde_sim["y min"] = ymm[0]
        fde_sim["y max"] = ymm[1]
        fde_sim["z min"] = -10e-6
        fde_sim["z max"] = 3e-6 


        # defining boundary conditions. It al
        fde_sim["y min bc"] = "PML"
        fde_sim["y max bc"] = "PML"
        fde_sim["z min bc"] = "PML"
        fde_sim["z max bc"] = "PML"
        
        fde_sim["mesh cells y"] = 50
        fde_sim["mesh cells z"] = 50


    def runsimulation(self):

        self.mode.run()

    def find_modes(self, wavl, trial_modes):
        self.mode.switchtolayout()
        self.mode.setnamed("FDE", "wavelength", wavl)
        self.mode.setanalysis("number of trial modes", trial_modes)
        return self.mode.findmodes()

    def plotmode(self, m, wavl,trial_modes):
        
        self.find_modes(wavl, trial_modes)
        
        y = self.mode.getdata(f"FDE::data::mode{m}", "y")
        z = self.mode.getdata(f"FDE::data::mode{m}", "z")
        plt.rcParams.update({'font.size': 17})
        
        y_vals = y * 10**6
        z_vals = z * 10**6

        E_x_abs = np.sqrt(np.abs(self.mode.getdata(f"FDE::data::mode{m}", "Ex")**2))
        E_y_abs = np.sqrt(np.abs(self.mode.getdata(f"FDE::data::mode{m}", "Ey")**2))
        E_z_abs = np.sqrt(np.abs(self.mode.getdata(f"FDE::data::mode{m}", "Ez")**2))

        H_x_abs = np.sqrt(np.abs(self.mode.getdata(f"FDE::data::mode{m}", "Hx")**2))
        H_y_abs = np.sqrt(np.abs(self.mode.getdata(f"FDE::data::mode{m}", "Hy")**2))
        H_z_abs = np.sqrt(np.abs(self.mode.getdata(f"FDE::data::mode{m}", "Hz")**2))

        E = E_x_abs**2 + E_y_abs**2  + E_z_abs**2
        H = H_x_abs**2 + H_y_abs**2  + H_z_abs**2
        E_dens = 1/2*(eps0*E + mu0*H) #lumerical mode is neglecting the 1/2 in energy density plot
        
        '''
        We have to do this, because of the nonuniform grid from lumerical, a very easy way is a simple interpolation
        to linearize E, H .. data for example.
        '''
        yy, zz = np.meshgrid(y_vals, z_vals, indexing='ij')

        yy_flat = yy.flatten()
        zz_flat = zz.flatten()

        grid_y, grid_z = np.mgrid[y_vals.min():y_vals.max():500j, z_vals.min():z_vals.max():500j]
        labels = ['E_x', 'E_y', 'E_z', 'H_x', 'H_y', 'H_z', 'E', 'H', 'Energy_dens']
        evything = [E_x_abs, E_y_abs, E_z_abs, H_x_abs, H_y_abs, H_z_abs, E, H, E_dens]
        
        for i in range(len(labels)):
            f_flat = evything[i].flatten()

            interp = griddata((yy_flat, zz_flat), f_flat, (grid_y, grid_z), method='linear')
            plt.figure(i)
            plt.imshow(interp.T, extent=[y_vals.min(), y_vals.max(), z_vals.min(), z_vals.max()],
                       origin='lower', aspect='equal', cmap='RdBu')
        

            plt.colorbar(label='Field intensity')
            plt.plot(np.array([self.mode.getnamed("sin_wg", 'y min'), self.mode.getnamed("sin_wg", 'y min'), 
                  self.mode.getnamed("sin_wg", 'y max'), self.mode.getnamed("sin_wg", 'y max'), 
                  self.mode.getnamed("sin_wg", 'y min')])*10**(6), 
                 np.array([self.mode.getnamed("sin_wg", 'z min'), self.mode.getnamed("sin_wg", 'z max'), 
                           self.mode.getnamed("sin_wg", 'z max'), self.mode.getnamed("sin_wg", 'z min'), 
                           self.mode.getnamed("sin_wg", 'z min')])*10**(6), color='black', lw=2)
            plt.axhline(self.mode.getnamed("tfln", 'z min')*10**(6))
            plt.axhline(self.mode.getnamed("tfln", 'z max')*10**(6))
            plt.xlabel('y (µm)')
            plt.ylabel('z (µm)')
            plt.title(f'Field Distribution {labels[i]}')
            plt.tight_layout()
        plt.show()
        
    
    def filtered_modes(self, pol_thres, pol):
        
        if pol == 'TE':
            mode_ids = [s.split("::")[2] for s in self.mode.getresult().split('\n') if 'mode' in s]
            return [i for i in mode_ids if self.mode.getdata(i, pol+" polarization fraction") > pol_thres]
        
        if pol == 'TM':
            pol = 'TE'
            mode_ids = [s.split("::")[2] for s in self.mode.getresult().split('\n') if 'mode' in s]
            return [i for i in mode_ids if self.mode.getdata(i, pol+" polarization fraction") < 1 - pol_thres]
 
    def select_mode(self, mode_id):
        
        self.mode.selectmode(mode_id)
        
    def dim_sweep(self, wavl, wavlsweep = 0, N = 5):
        '''
        this can be used to change the dimensions of one specific structure (e.g. the waveguide)#
        and so get neff vs. some spatial dimension
        '''
        
        '''
        TODO: adjust the mesh, so just delete and make new mesh at boundary of wg + 0.2 or so at each step
        '''
        sin_wg = 'sin_wg'
        material = 'Si3N4 (Silicon Nitride) - Luke'
        
        ymins = np.linspace(-5, -0.1, N)*10**(-6)
        ymaxs = np.linspace(5, 0.1, N)*10**(-6)
        
        zborder = 0.28*10**(-6)
        zmaxs = np.linspace(1.5, 0.3, N)*10**(-6)
        
        neffsTE = np.zeros([3, N])
        neffsTM = np.zeros([3, N])
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = timestr
       
        filename_details = "details_" + timestr
        with open(filename_details, "w") as text_file:
            text_file.write("Wavelength m: %s" % wavl)
            text_file.write("\n")
            text_file.write("Dimension sweep width m: %s" % [2*ymaxs[-1], 2*ymaxs[0], N])
            text_file.write("\n")
            text_file.write("Dimension sweep height (upper border) m: %s" % [zmaxs[-1], zmaxs[0], N])
            text_file.write("\n")
            text_file.write("Dimension sweep height (lower border) m: %s" % zborder)
            text_file.write("\n")
            text_file.write("Material: %s" % material)
        
        for j in range(len(zmaxs)):
            for i in range(len(ymins)):
                print((i+j)/(2*N)*100, '%')
                self.mode.switchtolayout()
                self.addmesh([ymins[i] - 0.5*10**(-6), ymaxs[i] + 0.5*10**(-6)], name = 'meshh')
                self.updatestructures(sin_wg, [ymins[i], ymaxs[i]], [zborder, zmaxs[j]], material)
                
                #self.updatestructures("saph_layer", [ymins[i]-5*10**(-6), ymaxs[i]+5*10**(-6)], [0.3*10**(-6), 1.1*10**(-6)], "Sapphire")
                #self.updatestructures("saph_layer", [ymins[i]-5*10**(-6), ymaxs[i]+5*10**(-6)], [0.3*10**(-6), 1.1*10**(-6)], 'Lithium niobate zcut')
                self.find_modes(wavl, 10)
                
                
                #here I filter out the first TE mode!
                mode_id = self.filtered_modes(0.96, "TE")
                
                try:
                    neffsTE[0, i] = [self.mode.getdata(mode_id[0], "neff")][0][0][0]
                    self.writedata(filename+"neff"+"te00", [[[self.mode.getdata(mode_id[0], "neff")][0][0][0]]])
                except IndexError:
                    print(f"Index 0 {i} out of range. Appending 0.")
                    neffsTE[0, i] = 0
                    self.writedata(filename+"neff"+"te00", [[0]])
    
                    
                try:
                    neffsTE[1, i] = [self.mode.getdata(mode_id[1], "neff")][0][0][0]
                    self.writedata(filename+"neff"+"te01", [[[self.mode.getdata(mode_id[1], "neff")][0][0][0]]])
                except IndexError:
                    print(f"Index 1 {i} out of range. Appending 0.")
                    neffsTE[1, i] = 0
                    self.writedata(filename+"neff"+"te01", [[0]])
    
                     
                try:
                    neffsTE[2, i] = [self.mode.getdata(mode_id[2], "neff")][0][0][0]
                    self.writedata(filename+"neff"+"te01", [[[self.mode.getdata(mode_id[2], "neff")][0][0][0]]])
                except IndexError:
                    print(f"Index 2 {i} out of range. Appending 0.")
                    neffsTE[2, i] = 0
                    self.writedata(filename+"neff"+"te02", [[0]])
    
    
                mode_id2 = self.filtered_modes(0.4, "TM")
                
                try:
                    neffsTM[0, i] = [self.mode.getdata(mode_id2[0], "neff")][0][0][0]
                    self.writedata(filename+"neff"+"tm00", [[[self.mode.getdata(mode_id2[0], "neff")][0][0][0]]])
                except IndexError:
                    print(f"TM Index 0 {i} out of range. Appending 0.")
                    neffsTM[0, i] = 0
                    self.writedata(filename+"neff"+"tm00", [[0]])
    
                    
                try:
                    neffsTM[1, i] = [self.mode.getdata(mode_id2[1], "neff")][0][0][0]
                    self.writedata(filename+"neff"+"tm01", [[[self.mode.getdata(mode_id2[1], "neff")][0][0][0]]])
                except IndexError:
                    print(f"TM Index 1 {i} out of range. Appending 0.")
                    neffsTM[1, i] = 0
                    self.writedata(filename+"neff"+"tm01", [[0]])
    
                     
                try:
                    neffsTM[2, i] = [self.mode.getdata(mode_id2[2], "neff")][0][0][0]
                    self.writedata(filename+"neff"+"tm02", [[[self.mode.getdata(mode_id2[2], "neff")][0][0][0]]])
                except IndexError:
                    print(f"TM Index 2 {i} out of range. Appending 0.")
                    neffsTM[2, i] = 0
                    self.writedata(filename+"neff"+"tm02", [[0]])
    
    
                #tmp = []
                #for j in range(len(mode_id2)):
                #    tmp.append([self.mode.getdata(mode_id2, "neff")][j])
    
                res = 0
                if wavlsweep == 1:
                    res = self.getbetas(3*10**(-6), 1.5*10**(-6), 10)
                
        return [neffsTM,neffsTE, res]
        
        
    def wavl_sweep(self, wavl_center, wavl_span, N_sweep, trial_modes=4, 
                pol_thres=0.96, pol="TE", mode_ind=0):
        wavls = []
        E_fields = []
        H_fields = []
        n_effs = []
        n_grps = []
        
        wavl_start = wavl_center - wavl_span
        wavl_stop = wavl_center + wavl_span
        lambdas =  np.linspace(wavl_start, wavl_stop, N_sweep)
        
        for wavl_i in lambdas:
            self.find_modes(wavl_i, trial_modes)
            mode_id = self.filtered_modes(pol_thres, pol)[mode_ind]
            self.select_mode(mode_id)
            E_field = [self.mode.getdata(mode_id, s)[:,:,0,0] 
                for s in ("Ex","Ey","Ez")]
            n_eff = [self.mode.getdata(mode_id, "neff")][0]
            H_field = [self.mode.getdata(mode_id, s)[:,:,0,0] 
                for s in ("Hx","Hy","Hz")]
            n_grp = [self.mode.getdata(mode_id, "ng")][0]
            E_fields.append(E_field)
            H_fields.append(H_field)
            n_effs.append(n_eff)
            n_grps.append(n_grp)
            wavls.append(wavl_i)
            
        xaxis = self.mode.getdata("FDE::data::material", "x")
        yaxis = self.mode.getdata("FDE::data::material", "y")
        index = self.mode.getdata("FDE::data::material", "index_y")
        
        c = 2.99792e-7
        lambdas =  np.linspace(wavl_start, wavl_stop, N_sweep) * 10**(9)
        omegas = 2*np.pi*c* lambdas
        n_effs = np.reshape(n_effs, len(n_effs))
        
        dneff = np.gradient(n_effs, lambdas) 
        d2neff = np.gradient(dneff, lambdas)


                
        D = - lambdas/c * d2neff
        
        #plt.plot(lambdas[2:-2], D[2:-2])
        #plt.xlabel("Wavelength (nm)")
        #plt.ylabel("Dispersion D (ps/nm/km)")
        #plt.title("Dispersion vs Wavelength")
        #plt.grid(True)
        #plt.show()
        
        
            
        return [xaxis, yaxis, index, wavls, np.array(E_fields), 
            np.array(H_fields), np.array(n_grps), np.array(n_effs)]
    
    def dispersion_sweep(self, wavl, maxheight, maxwidth, N, material):
        
        sin_wg = 'sin_wg'
        
        ymins = np.linspace(-maxheight, -0.25, N)*10**(-6)
        ymaxs = np.linspace(maxheight, 0.25, N)*10**(-6)
        
        xmins = np.linspace(-maxwidth, -0.1, N)*10**(-6)
        xmaxs = np.linspace(maxwidth, 0.1, N)*10**(-6)
        
        neffsTE00 = np.zeros([N, N])
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{material},dims{ymaxs[0]*10**(6)},{ymaxs[-1]*10**(6)}, time" + timestr
        
        for i in range(len(ymins)):
            for j in range(len(xmins)):
                
                print((i*j)/(len(xmins)*len(ymins)) *100, '%')
                
                self.mode.switchtolayout()
                self.addmesh([xmins[i] - 0.5*10**(-6), xmaxs[i] + 0.5*10**(-6)], name = 'meshh')
                self.updatestructures(sin_wg, [xmins[i], xmaxs[i]], [0.3*10**(-6), (0.3*10**(-6) + 2*ymaxs[j])], material)
                
                tmp = self.getbetas(wavl, 200*10**(-9), 30)
                
                try:
                    #neffsTE00[j, i] = [self.mode.getdata(mode_id[0], "neff")][0][0][0]
                    #neffsTE00[j, i] = [self.mode.getdata(mode_id[0], "neff")][0][0][0]

                    self.writedata(filename+"neff"+"te00", tmp)
                except IndexError:
                    print(f"Index 0 {i} out of range. Appending 0.")
                    neffsTE00[j, i] = 0
                    self.writedata(filename+"neff"+"te00", [[0]])
                           
    
    def getbetas(self, wavl, interval, N):
        
        tmp = self.wavl_sweep(wavl, interval, N)
        neffs = np.array(tmp[-1])
        ngs = np.array(tmp[-2])
        omega0 = 2*np.pi*c/wavl
                
        wavls = np.array(tmp[3])
        
        beta = 2*np.pi*neffs/wavls
        '''
        omega = 2*np.pi*c/wavls

        #Lets get the derivatives by making a spline and taking its derivative
        spline = UnivariateSpline(omega[::-1], beta[::-1], k=5, s=0)

        beta1 = spline.derivative(1)
        beta2 = spline.derivative(2)
        beta3 = spline.derivative(3)
        beta4 = spline.derivative(4)
        beta5 = spline.derivative(5)
        
        plt.figure()
        c_sc = 2.99792e-7
        omega_rescale = 2*np.pi*c_sc/(wavls* 10**(9))
        plt.plot(wavls, -omega**2/(2*np.pi*c)* beta2(omega))    
        res = [beta1(omega0), beta2(omega0), beta3(omega0), beta4(omega0), beta5(omega0)]
        '''
        return [beta]
    
    def writedata(self, name, value):
        with open(name, 'a') as f:
            csv.writer(f, delimiter=',').writerows(value)
            
    def effmodea(self, m=1, wavl=1.55*10**(-6)):
        
        ''' 
        TODO: Alternative way to do effmodearea but needs to BE FIXED TOO 
        Calculate the effective mode area using integration over the mode fields
        and material properties. 
        '''
        # Get the electric and magnetic field components for the mode m
        E2 = self.mode.pinch(self.mode.getelectric(f"FDE::data::mode{m}"))
        H2 = self.mode.pinch(self.mode.getmagnetic(f"FDE::data::mode{m}"))
        
        # Set the analysis for a slightly shifted wavelength for material properties
        self.mode.setanalysis('wavelength', wavl + 1e-9)
        self.find_modes(wavl, 4)
        
        tmp = "FDE::data::material"
        f1 = self.mode.getdata(tmp, "f")
        eps1 = self.mode.pinch(self.mode.getdata(tmp, "index_x"))**2

        z = self.mode.getdata(tmp, "z")
        y = self.mode.getdata(tmp, "y")

        # Shift the wavelength again to calculate for another point
        self.mode.setanalysis('wavelength', wavl - 1e-9)
        self.find_modes(wavl, 4)
        
        f3 = self.mode.getdata(tmp, "f")
        eps3 = self.mode.pinch(self.mode.getdata(tmp, "index_x"))**2

        # Calculate the real part of the derivative of epsilon with respect to frequency
        re_depsdw = np.real((f3 * eps3 - f1 * eps1) / (f3 - f1))

        # Calculate the power density (W) from the electric and magnetic fields
        W = 0.5 * (re_depsdw * eps0 * E2 + mu0 * H2)

        # Integrate over the z dimension and then over the y dimension
        integrated_z = np.trapz(W, z, axis=1)
        integrated_yz = np.trapz(integrated_z, y)

        # Normalize by the maximum value of W to get the effective mode area
        modalarea1 = integrated_yz / np.max(W)

        # Print the result
        print(f"Effective mode area\nmethod 1: {modalarea1} microm^2")
 
    def effmodea(self, m=1, wavl=1.55*10**(-6)):
        
        '''
        TODO: NEEDS TO BE FIXED
        '''
        E2 = self.mode.pinch(self.mode.getelectric(f"FDE::data::mode{m}"))
        H2 = self.mode.pinch(self.mode.getmagnetic(f"FDE::data::mode{m}"))

        self.mode.setanalysis('wavelength',wavl+1e-9)
        self.find_modes(wavl, 4)
        
        tmp = "FDE::data::material"
        f1 = self.mode.getdata(tmp,"f")
        eps1 = self.mode.pinch(self.mode.getdata(tmp, "index_x"))**2
                
        z = self.mode.getdata(tmp,"z")
        y = self.mode.getdata(tmp,"y")
        
        self.mode.setanalysis('wavelength',wavl-1e-9)
        self.find_modes(wavl, 4)

        f3 = self.mode.getdata(tmp,"f")
        eps3 = self.mode.pinch(self.mode.getdata(tmp, "index_x"))**2


        re_depsdw = np.real((f3*eps3-f1*eps1)/(f3-f1))
        
        W = 0.5*(re_depsdw*eps0*E2 + mu0*H2)
        
                
        integrated_z = np.trapz(W, z, axis=1)
        integrated_yz = np.trapz(integrated_z, y)

        modalarea1 = integrated_yz / np.max(W)

        print("Effective mode area\n")
        print("method 1: " +(modalarea1)+"microm^2")

        return W
    
    def supercontinuum(self, wavl):
        
        
        betas = np.array(self.getbetas(1.7*10**(-6), 0.2*10**(-6)), 30)
        d = betas[-1]
        betas = betas[:-1]
        setup = gnlse.GNLSESetup()

        # Numerical parameters
        setup.resolution = 2**14
        setup.time_window = 12.5  # ps
        setup.z_saves = 200

        # Physical parameters
        setup.wavelength = 1700  # nm
        setup.fiber_length = 0.15  # m
        setup.nonlinearity = 0.11  # 1/W/m
        setup.raman_model = gnlse.raman_blowwood
        setup.self_steepening = True

        # The dispersion model is built from a Taylor expansion with coefficients
        # given below.
        loss = 0
        
        setup.dispersion_model = gnlse.DispersionFiberFromTaylor(loss, betas)

        # Input pulse parameters
        peak_power = 10000  # W
        duration = 0.050  # ps

        # This example extends the original code with additional simulations for
        pulse_model = gnlse.SechEnvelope(peak_power, duration)
        

        plt.figure(figsize=(14, 8), facecolor='w', edgecolor='k')
        print('%s...' % pulse_model)

        setup.pulse_model = pulse_model
        solver = gnlse.GNLSE(setup)
        solution = solver.run()

        plt.figure()
        plt.title(pulse_model.name)
        gnlse.plot_wavelength_vs_distance(solution, WL_range=[setup.wavelength-400, setup.wavelength+400])

        plt.figure()
        gnlse.plot_delay_vs_distance(solution, time_range=[-0.5, 5])

        plt.tight_layout()
        plt.show()
        
        return([betas, d])
    
    def savemode(self):
        self.mode.save()

    def stop(self):
        self.mode.close()
        
if __name__ == "__main__":
    
    run1 = lumerical_sim()
    run1.addmaterial()
    run1.addstructures()
    run1.addsimulation()
    res2 = run1.dim_sweep(3.3*10**(-6), 0, 50)
    res3 = run1.dim_sweep(1.55*10**(-6), 0, 50)
    res4 = run1.dim_sweep(4*10**(-6), 0, 50)

    #res = run1.getbetas(1.7*10**(-6), 0.2*10**(-6), 10)
    #run1.supercontinuum()
    #neffs = run1.dim_sweep(1.55*10**(-6))
    #plt.figure()
    #plt.plot(np.reshape(neffs, len(neffs)))
    #run1.plotmode(1, 1.55*10**(-6), 10)
    #run1.effmodea()
    #run1.run_sweep(2*10**(-6), 1*10**(-6), 20)
#%%
run1.addstructures()
run1.addmesh([-1*10**(-6), 1*10**(-6)], 'temp')
run1.addsimulation()
res2 = run1.dim_sweep(3.3*10**(-6), 0, 50)
#res2 = run1.dispersion_sweep(3.6*10**(-6), 0.8, 5, 50, 'Si3N4 (Silicon Nitride) - Luke')
#%%
plt.title('3300: Sa-300nmSiN-800nmAl2O3n=1.46')
plt.rcParams["font.size"] = '18'

#tm = np.array([x.item() if isinstance(x, np.ndarray) else x for x in res1[0]])
#tm = tm[tm != 0]


#te = np.array([x.item() if isinstance(x, np.ndarray) else x for x in res1[1]])

widths2 = np.linspace(5, 1, len(res2[0][0]))*2

plt.plot(widths2[res2[1][0] != 0], res2[1][0][res2[1][0] != 0], color = 'blue', label = 'TE00')
plt.plot(widths2[res2[1][0] != 0][1::2 - 1], res2[1][0][res2[1][0] != 0][1::2 - 1], 'bo')

plt.plot(widths2[res2[1][1] != 0], res2[1][1][res2[1][1] != 0], color = 'lightblue', label = 'TE01')
plt.plot(widths2[res2[1][1] != 0][1::2 - 1], res2[1][1][res2[1][1] != 0][1::2 - 1], 'o', color = 'lightblue')

#plt.plot(widths2[res2[1][2] != 0], res2[1][2][res2[1][2] != 0], color = 'darkblue', label = 'TE02')
#plt.plot(widths2[res2[1][2] != 0][1::2], res2[1][2][res2[1][2] != 0][1::2], 'o', color = 'darkblue')


plt.plot(widths2[res2[0][0] != 0], res2[0][0][res2[0][0] != 0], color = 'red', label = 'TM00')
plt.plot(widths2[res2[0][0] != 0][1::2-1], res2[0][0][res2[0][0] != 0][1::2-1], 'ro')

diff1 = np.abs(np.diff(res2[0][1]))
diff2 = np.abs(np.diff(res2[0][2]))
threshold = 0.004



mask1 = np.insert(diff1 < threshold, 0, True) & np.append(diff1 < threshold, True)
mask2 = np.insert(diff2 < threshold, 0, True) & np.append(diff2 < threshold, True)

tm01 = res2[0][1][mask1]
tm02 = res2[0][2][mask2]


#plt.plot(widths2[mask1][tm01 != 0], tm01[tm01 != 0], color = 'orangered', label = 'TM01')
plt.plot(widths2[mask1][tm01 != 0][1::2], tm01[tm01 != 0][1::2], 'o', color = 'orangered')

#plt.plot(widths2[mask2][tm02 != 0], tm02[tm02 != 0], color = 'darkred', label = 'TM02')
plt.plot(widths2[mask2][tm02 != 0][1::2], tm02[tm02 != 0][1::2], 'o', color = 'darkred')

plt.legend(fontsize = 10)
plt.grid()
plt.xlabel('width in [$\mu$m]')
plt.ylabel('Effective index')
plt.xlim([0,10])
plt.savefig('3300SINAl2O3_300nmsin2p2_andleakage1p46_2.pdf', dpi = 1000, bbox_inches = 'tight')