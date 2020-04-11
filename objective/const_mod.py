import sys 
import logging 

# Gas Constants ( J/mol K ): 

R = 8.314462618 

# Avogadro constant:  

NA = 6.02214179*10**(23) 

# Boltzman Constants (J/K) : 

kb = 1.38064852*10**(-23) 

# Water's Molar mass:

water_mol_mass = 18.01528  

# Convert LAMMPS types of units into SI  

class Units(): 


    def __init__(self,keyword):  

        self.units_logger = logging.getLogger(__name__)  

        units_list = { 

        "real": "RealToSI",  
        "metal": "MetaToSI"

        } 
    
        existence = units_list.get(keyword) 
        
        if ( not existence): 
        
            self.units_logger.error( "Units: %s not defined in const_mod.py"%keyword  ) 

            sys.exit()

        else:
                
            getunits = units_list[keyword]  
        
            unitsname = getattr(self,getunits) 

            unitsname()  

    def RealToSI(self):  

        # Distance ( 1 Angstroms = 10^-10 m ): 

        self.dist_scale = 10**(-10)  

        # Volume (  1 A^3 = 10^(-30) m^3 )  

        self.vol_scale = 10**(-30)  

        # Pressure ( 1 atmosphere = 101325 Pascal )  

        self.p_scale = 101325  
            
        # Energy ( 1 Kcal/mole = 4184 J/mole): 
        
        self.e_scale = 4184 

        # Time ( 1 femtoseconds = 10^-15 s ) 

        self.t_scale = 10**(-15) 
    
        # Force ( Kcal/mol-Angstrom) 

        self.f_scale = self.e_scale/(NA*self.dist_scale)


    def MetalToSI(self): 

        # Distance ( 1 Angstroms = 10^-10 m ): 

        self.dist_scale = 10**(-10)  

        # Volume (  1 A^3 = 10^(-30) m^3 )  

        self.vol_scale = 10**(-30)  

        # Pressure ( 1 bar = 100,000 Pascal )  

        self.p_scale = 100000  
            
        # Energy ( 1 ev = 96485 J/mole): 
        
        self.e_scale = 96485 

        # Time ( 1 picosecond = 10^-12 s ) 

        self.t_scale = 10**(-12) 

        # Force ( 1ev/Angstroms = J/m  ) 

        self.f_scale =  self.e_scale/(NA*self.dist_scale) 



