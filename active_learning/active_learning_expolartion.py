from ase.io import read, write
from ase.build import make_supercell, sort
from ase.constraints import ExpCellFilter,StrainFilter
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.io.trajectory import Trajectory,TrajectoryReader
from ase.md import MDLogger
from ase.optimize.sciopt import SciPyFminCG
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary
from ase import units
from glob import glob
import numpy as np
import json
import os



# jdata dict format:
# Pot_name: Nequip, Deepmd
# Pot_list: (000.pb...003.pb), (000.pth...004.pth)
# Pot: pot_potential
# if Nequip: THREADS
# log_inverval
# lower_bound: 0.1
# upper_bound: 0.5
# Ensemble：NVT; NPT; strain_search
# NVT
# Params:{Name, Seed, T(K), atom, strain_x, strain_y, theta(degree), step, time_step(fs)}
# NPT
# Params:{Name, Seed, T(K), atom, step, time_step(fs)}
# strain_search
# Params:{Name, Seed, T(K), atom, strain_x_range[example: (0.00,0.30,0.01)], 
#                           mu[example:0.2], strain_y_increase[0.01],
#                           theta(degree), strain_rate(fs^-1), tension_interval,time_step(fs)}
# uniaxial_tension
# Params:{Name, Seed, T(K), atom, direction(x,y), strain_max(0.31), strain_rate(fs^-1), 
#                           tension_interval, time_step(fs)}


def exploration_one(jdata):
    pot = [] 
    if jdata['Pot_name'] == 'Nequip':
        from nequip.ase.nequip_calculator import nequip_calculator
        os.environ['OMP_NUM_THREADS'] = jdata['THREADS']
        for p in jdata['Pot_list']:
            pot.append(nequip_calculator(p))
    elif jdata['Pot_name'] == 'Deepmd':
        from deepmd.calculator import DP
        file = sorted(glob(os.path.join(jdata['work_dir'],'model/*.pb')))
        jdata['Pot_list'] = file
        for p in jdata['Pot_list']:
            pot.append(DP(p))
    jdata['Pot'] = pot
    
    if jdata['Ensemble'] == 'NVT':
        NVT_run(jdata)
    elif jdata['Ensemble'] == 'NPT':
        NPT_run(jdata)
    elif jdata['Ensemble'] == 'strain_search':
        strain_search_run(jdata)
    elif jdata['Ensemble'] == 'uniaxial_tension':
        uniaxial_tension(jdata)

def uniaxial_tension(jdata):
    np.random.seed(jdata['Params']['Seed'])
    MD_atom = read(jdata['Params']['atom'])
    strain_max = jdata['Params']['strain_max']
    
    strain_step = jdata['Params']['strain_rate'] * jdata['Params']['time_step'] # strain/step
    tension_interval = jdata['Params']['tension_interval']
    
    if os.path.exists(jdata['Params']['Name']+'.log'):
        os.remove(jdata['Params']['Name']+'.log')
    
    index = [0]
    candicate_list = [] 
    failed_list = []
    global_step = 0
    
    new_atom = MD_atom.copy()
    new_atom.set_calculator(jdata['Pot'][0])
    cell     = np.array(new_atom.get_cell()).copy()
    MaxwellBoltzmannDistribution(new_atom, temperature_K=jdata['Params']['T'])
    Stationary(new_atom)
    dyn = Langevin(new_atom, timestep=jdata['Params']['time_step']*units.fs,
                       temperature_K=jdata['Params']['T'], friction=0.2)
    
    dyn.attach(MDLogger(dyn, new_atom, jdata['Params']['Name']+'.log', header=True, stress=False,
                peratom=False, mode="a"), interval=jdata['log_inverval'])
    
    dyn.attach(uncertain_estimate, interval=1,
                   dyn=dyn, MD_atom=new_atom, jdata=jdata, 
                   candicate_list=candicate_list,failed_list=failed_list,
                   index=index, global_step=global_step)
    
    
    def tension_local(dyn,new_atom, strain_step, theta, cell):
        step = dyn.get_number_of_steps()
        R = rotate_matirx(theta*np.pi/180)
        # current_cell = np.array(new_atom.get_cell())
        F = np.array([[1+strain_step*step,0],[0, 1]])
        F = rotate_tensor(F,R.T)
        F_3D = np.diag([1.,1.,1.])
        F_3D[:2,:2] = F
        current_cell = np.matmul(cell,F_3D)
        new_atom.set_cell(current_cell,scale_atoms=True)
    
    
    if jdata['Params']['direction'] == 'x':
        dyn.attach(tension_local,interval=tension_interval,dyn=dyn,new_atom=new_atom,
                      strain_step=strain_step,theta=0,cell=cell)
    elif jdata['Params']['direction'] == 'y':
        dyn.attach(tension_local,interval=tension_interval,dyn=dyn,new_atom=new_atom,
                      strain_step=strain_step,theta=90,cell=cell)
    
    step = np.ceil(strain_max/strain_step)
    dyn.run(step)
    
    write_estimate(candicate_list,failed_list,jdata)
    
        
        
def strain_search_run(jdata):
    np.random.seed(jdata['Params']['Seed'])
    MD_atom = read(jdata['Params']['atom'])
    theta = jdata['Params']['theta']
    
    strain_step = jdata['Params']['strain_rate'] * jdata['Params']['time_step'] # strain/step
    tension_interval = jdata['Params']['tension_interval']
    
    FX = np.arange(jdata['Params']['strain_x_range'][0],
                   jdata['Params']['strain_x_range'][1],
                   jdata['Params']['strain_x_range'][2]) + 1
    
    mu = jdata['Params']['mu']
    
    if os.path.exists(jdata['Params']['Name']+'.log'):
        os.remove(jdata['Params']['Name']+'.log')
    
    index = [0]
    candicate_list = [] 
    failed_list = []
    global_step = 0
    for F_x in FX:
        new_atom = MD_atom.copy()
        new_atom.set_calculator(jdata['Pot'][0])
        cell     = np.array(new_atom.get_cell()).copy()
        R = rotate_matirx(theta*np.pi/180)
        F = np.array([[F_x, 0], [0, 1-(F_x-1)*mu] ])
        F = rotate_tensor(F,R.T)
        F_3D = np.diag([1.,1.,1.])
        F_3D[:2,:2] = F # added
        new_cell = np.matmul(cell,F_3D)
        new_atom.set_cell(new_cell,scale_atoms=True)
        
        MaxwellBoltzmannDistribution(new_atom, temperature_K=jdata['Params']['T'])
        Stationary(new_atom)
        
        dyn = Langevin(new_atom, timestep=jdata['Params']['time_step']*units.fs,
                       temperature_K=jdata['Params']['T'], friction=0.2)
        
        with open(jdata['Params']['Name']+'.log','a') as f:
            f.write('strainx: %.2f\n'%(F_x))
            
        dyn.attach(MDLogger(dyn, new_atom, jdata['Params']['Name']+'.log', header=True, stress=False,
                   peratom=False, mode="a"), interval=jdata['log_inverval'])
        dyn.attach(uncertain_estimate, interval=1,
                   dyn=dyn, MD_atom=new_atom, jdata=jdata, 
                   candicate_list=candicate_list,failed_list=failed_list,
                   index=index, global_step=global_step)
        
        
        dyn.attach(tension,interval=tension_interval,dyn=dyn,new_atom=new_atom,
                  strain_step=strain_step,theta=theta,lamda=F_x,mu=mu,cell=cell)
        
        step = np.ceil((F_x - (1-(F_x-1)*mu) + jdata['Params']['strain_y_increase'])/strain_step)
        dyn.run(step)
        global_step += step + 1
    
    write_estimate(candicate_list,failed_list,jdata)
        
        
def tension(dyn,new_atom,strain_step, theta, lamda, mu, cell):
    step = dyn.get_number_of_steps()
    R = rotate_matirx(theta*np.pi/180)
    # current_cell = np.array(new_atom.get_cell())
    F = np.array([[lamda,0],[0, (1-(lamda-1)*mu) + strain_step*step]])
    F = rotate_tensor(F,R.T)
    F_3D = np.diag([1.,1.,1.])
    F_3D[:2,:2] = F
    current_cell = np.matmul(cell,F_3D)
    new_atom.set_cell(current_cell,scale_atoms=True)
        
    
def NVT_run(jdata):
    np.random.seed(jdata['Params']['Seed'])
    theta = jdata['Params']['theta']
    MD_atom = read(jdata['Params']['atom'])
    
    cell = MD_atom.get_cell()
    R = rotate_matirx(theta*np.pi/180)
    F = np.array([[jdata['Params']['strain_x']+1,0],
                  [0,jdata['Params']['strain_y']+1]])
    F = rotate_tensor(F,R.T)
    F_3D = np.diag([1.,1.,1.])
    F_3D[:2,:2] = F
    tmp_cell = np.matmul(cell,F_3D)
    
    R_2_3D = np.diag([1.,1.,1.])
    R_2_3D[:2,:2] = rotate_matirx(np.arctan(tmp_cell[1][0]/tmp_cell[1][1]))
    
    tmp_2_cell = np.matmul(tmp_cell,R_2_3D.T)
    if np.abs(tmp_2_cell[1,0])<1e-10:
        tmp_2_cell[1,0] = 0
    else:
        raise NotImplementedError('cell[1,0] not equal to 0 !')
    MD_atom.set_cell(tmp_2_cell,scale_atoms=True)
    MD_atom.set_calculator(jdata['Pot'][0])
    MaxwellBoltzmannDistribution(MD_atom,
                                 temperature_K=jdata['Params']['T'],
                                 rng=np.random)
    Stationary(MD_atom)
    dyn = Langevin(MD_atom, timestep=jdata['Params']['time_step']*units.fs,
                       temperature_K=jdata['Params']['T'], friction=0.2)
    
    
    if os.path.exists(jdata['Params']['Name']+'.log'):
        os.remove(jdata['Params']['Name']+'.log')
    
    index = [0]
    candicate_list = [] 
    failed_list = []
    dyn.attach(MDLogger(dyn, MD_atom, jdata['Params']['Name']+'.log', header=True, stress=False,
               peratom=False, mode="a"), interval=jdata['log_inverval'])
    dyn.attach(uncertain_estimate, interval=1,dyn=dyn, MD_atom=MD_atom,
               jdata=jdata,candicate_list=candicate_list,
               failed_list=failed_list,index=index)
    dyn.run(jdata['Params']['Step'])
    
    write_estimate(candicate_list,failed_list,jdata)
    
    

def NPT_run(jdata):
    np.random.seed(jdata['Params']['Seed'])
    
    MD_atom = read(jdata['Params']['atom'])
    MD_atom.set_calculator(jdata['Pot'][0])
    cell = MD_atom.get_cell()
    if np.abs(cell[1,0])<1e-10:
        cell[1,0] = 0
    if np.abs(cell[2,0])<1e-10:
        cell[2,0] = 0
    if np.abs(cell[2,1])<1e-10:
        cell[2,1] = 0
    MD_atom.set_cell(cell,scale_atoms=True)
    MaxwellBoltzmannDistribution(MD_atom,
                                 temperature_K=jdata['Params']['T'],
                                 rng=np.random)
    Stationary(MD_atom)
    dyn = NPT(MD_atom, timestep=jdata['Params']['time_step']*units.fs, 
              temperature_K=jdata['Params']['T'], externalstress=np.zeros((3,3)),
              ttime=5*units.fs, pfactor=6, mask=[[1,1,0],[1,1,0],[0,0,0]])
    
    
    if os.path.exists(jdata['Params']['Name']+'.log'):
        os.remove(jdata['Params']['Name']+'.log')
    
    index = [0]
    candicate_list = [] 
    failed_list = []
    dyn.attach(MDLogger(dyn, MD_atom, jdata['Params']['Name']+'.log', header=True, stress=False,
               peratom=False, mode="a"), interval=jdata['log_inverval'])
    dyn.attach(uncertain_estimate, interval=1,dyn=dyn, MD_atom=MD_atom,
               jdata=jdata,candicate_list=candicate_list,
               failed_list=failed_list,index=index)
    dyn.run(jdata['Params']['Step'])
    
    write_estimate(candicate_list,failed_list,jdata)
    
    

def rotate_matirx(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])


def rotate_tensor(F, R):
    return np.matmul(np.matmul(R.T,F),R)


def write_estimate(candicate_list,failed_list,jdata):
    if (len(candicate_list) == 0):
        with open(jdata['Params']['Name']+'.out', 'r+') as f:
            content = f.read()
            f.seek(0,0)
            f.write(f'#Summery: selected: {0} unselected: {0} failed: {len(failed_list)}\n'+content)
    else:
        candicate_step = np.diff( [st.info['Step'] for st in candicate_list])

        select = np.argwhere(candicate_step != 1).flatten() + 1
        select = np.insert(select,0,0)

        with open(jdata['Params']['Name']+'.out', 'r+') as f:
            content = f.read()
            f.seek(0,0)
            f.write(f'#Summery: selected: {len(select)} unselected: {len(candicate_list)-len(select)} failed: {len(failed_list)}\n'+content)
       
    if len(failed_list) != 0:
        if os.path.exists(jdata['Params']['Name']+'_failed.xyz'):
            os.remove(jdata['Params']['Name']+'_failed.xyz')
        write(jdata['Params']['Name']+'_failed.xyz',failed_list,
              format='extxyz',columns=['symbols', 'positions'],
       write_results=False,write_info=True)
    if len(candicate_list) != 0:
        if os.path.exists(jdata['Params']['Name']+'_candicate.xyz'):
            os.remove(jdata['Params']['Name']+'_candicate.xyz')
        write(jdata['Params']['Name']+'_candicate.xyz',candicate_list,
              format='extxyz',columns=['symbols', 'positions'],
       write_results=False,write_info=True)
        

def uncertain_estimate(dyn, MD_atom, jdata,candicate_list,failed_list,index,global_step=0):
    # global index, candicate_list, failed_list
    
    lower_bound = jdata['lower_bound']
    upper_bound = jdata['upper_bound']
    forces = [MD_atom.get_forces()]
    energies = [MD_atom.get_potential_energy()]
    
    for p in jdata['Pot'][1:]:
        p.calculate(MD_atom)
        forces.append(p.results['forces'])
        energies.append(p.results['energy'])
        
    forces = np.array(forces)
    energies = np.array(energies)

    energies_uncertain = np.max(energies) - np.min(energies)
    forces_uncertain = np.max(np.max(forces,axis=0)-np.min(forces,axis=0))
    
    
    if index[0] == 0:
        if os.path.exists(jdata['Params']['Name']+'.out'):
            os.remove(jdata['Params']['Name']+'.out')
        f = open(jdata['Params']['Name']+'.out','a')
        f.write('  Step    Max_dev_energy  Max_dev_forces   T        Use\n')
        f.close()
        index[0] += 1
        
    if (forces_uncertain>lower_bound) and (forces_uncertain<upper_bound):
        f = open(jdata['Params']['Name']+'.out','a')
        f.write('%5d %14.5f  %14.5f  %9.3f    True\n'%(dyn.get_number_of_steps()+global_step,
                                                     energies_uncertain,
                                                     forces_uncertain,
                                                     MD_atom.get_temperature()))
        f.close()
        tmp_atom = MD_atom.copy()
        tmp_atom.info = { 'Step':dyn.get_number_of_steps()+global_step,
                          'energies_uncertain':energies_uncertain,
                          'forces_uncertain':forces_uncertain,
                          'T':MD_atom.get_temperature()}
        candicate_list.append(tmp_atom)
    elif  (forces_uncertain>upper_bound):
        f = open(jdata['Params']['Name']+'.out','a')
        f.write('%5d %14.5f  %14.5f  %9.3f    False\n'%(dyn.get_number_of_steps()+global_step,
                                                     energies_uncertain,
                                                     forces_uncertain,
                                                     MD_atom.get_temperature()))
        f.close()
        tmp_atom = MD_atom.copy()
        tmp_atom.info = { 'Step':dyn.get_number_of_steps()+global_step,
                          'energies_uncertain':energies_uncertain,
                          'forces_uncertain':forces_uncertain,
                          'T':MD_atom.get_temperature()}
        failed_list.append(tmp_atom) 
        

        
        
"""
Templete data dict:
data = {'work_dir': 'work'
        'Pot_name':'Deepmd',
         'Pot_list':['000.pb','001.pb','002.pb','003.pb'],
         'THREADS':'24',
         'log_inverval':100,
         'lower_bound':0.05,
         'upper_bound':0.2,
         'exploration':
        [
             {
                 'Ensemble':'strain_search',
                 'Params':{
                     'Name':'strain_search_test','Seed':974647,'T':10,
                     'atom':'B36N36.vasp', 'time_step':1,
                     'strain_x_range':[0.00,0.20,0.01],
                     'mu':0.2, 'strain_y_increase':0.01,
                     'theta':[0,3,6,9,12,15,18,21,24,27,30], 'strain_rate':1e-5,'tension_interval':10
                  }
             },
             
             {
                 'Ensemble':'NPT',
                 'Params':{
                     'Name':'NPT_test','Seed':974647,'T':10,
                     'atom':'B36N36.vasp','Step':1000, 'time_step':1
                  }
             },
             
             {
                 'Ensemble':'NVT',
                 'Params':{'Name':'NVT_test','Seed':974647,'T':15,
                   'atom':'B36N36.vasp','strain_x':[0.15,0.16,0.17],'strain_y':[-0.01,-0.02,-0.03],
                   'theta':[15,16,17], 'Step':1000, 'time_step':1
                  }
             },
             
             {
                 'Ensemble':uniaxial_tension,
                 'Params':{'Name':'uniaxial_tension', 'Seed':974647,'T':15,
                 'atom':[atom1,atom2,atom3,...],direction=['y','y',...], strain_max:0.31,
                 'strain_rate':1e-5,'tension_interval':10,'time_step':1}
             }
             
             
         ]
       } 

"""


def split_dict(data):
    exploration = data.pop('exploration')
    from deepmd.calculator import DP
    file = sorted(glob(os.path.join(data['work_dir'],'model/*.pb')))
    data['Pot_list'] = file
    data['Pot_list'] = [os.path.abspath(p) for p in data['Pot_list']]
    
    split_data = []
    for exp in exploration:
        if exp['Ensemble'] == 'strain_search':
            if type(exp['Params']['theta']) == list:
                theta = exp['Params']['theta']
                for i, th in enumerate(theta):
                    tmp_data = data.copy()
                    tmp_data['Ensemble'] = 'strain_search'
                    tmp_data['Params'] = exp['Params'].copy()
                    tmp_data['Params']['theta'] = th
                    tmp_data['Params']['Name'] =  tmp_data['Params']['Name']+f'_{th}'
                    tmp_data['Params']['Seed'] += i
                    split_data.append(tmp_data)

            elif (type(exp['Params']['theta']) == int) and (type(exp['Params']['theta']) == float):
                th = exp['Params']['theta']
                tmp_data = data.copy()
                tmp_data['Ensemble'] = 'strain_search'
                tmp_data['Params'] = exp['Params'].copy()
                tmp_data['Params']['Name'] = tmp_data['Params']['Name']+f'_{th}'
                split_data.append(tmp_data)
                
        elif exp['Ensemble'] == 'uniaxial_tension':
            if type(exp['Params']['atom']) == list:
                atoms = exp['Params']['atom']
                directions = exp['Params']['direction']
                for i, at in enumerate(atoms):
                    tmp_data = data.copy()
                    tmp_data['Ensemble'] = 'uniaxial_tension'
                    tmp_data['Params'] = exp['Params'].copy()
                    tmp_data['Params']['atom'] = at
                    tmp_data['Params']['Name'] = tmp_data['Params']['Name']+f'_{i}'
                    tmp_data['Params']['direction'] = directions[i]
                    tmp_data['Params']['Seed'] += i
                    split_data.append(tmp_data)
            elif (type(exp['Params']['atom']) == str):
                tmp_data = data.copy()
                tmp_data['Ensemble'] = 'uniaxial_tension'
                tmp_data['Params'] = exp['Params'].copy()
                tmp_data['Params']['Name'] = tmp_data['Params']['Name']+f'_{0}'
                split_data.append(tmp_data)

        elif exp['Ensemble'] == 'NPT':
            tmp_data = data.copy()
            tmp_data['Ensemble'] = 'NPT'
            tmp_data['Params'] = exp['Params'].copy()
            split_data.append(tmp_data)


        elif exp['Ensemble'] == 'NVT':
            if type(exp['Params']['theta']) == list:
                theta = exp['Params']['theta']
                strain_x = exp['Params']['strain_x']
                strain_y = exp['Params']['strain_y']

                for i in range(len(theta)):
                    tmp_data = data.copy()
                    tmp_data['Ensemble'] = 'NVT'
                    tmp_data['Params'] = exp['Params'].copy()
                    tmp_data['Params']['strain_x'] = strain_x[i]
                    tmp_data['Params']['strain_y'] = strain_y[i]
                    tmp_data['Params']['theta'] = theta[i]
                    tmp_data['Params']['Seed'] += i
                    tmp_data['Params']['Name'] = tmp_data['Params']['Name'] + f'_{i}'
                    split_data.append(tmp_data)

            elif (type(exp['Params']['theta']) == int) and (type(exp['Params']['theta']) == float):
                tmp_data = data.copy()
                tmp_data['Ensemble'] = 'strain_search'
                tmp_data['Params'] = exp['Params'].copy()
                tmp_data['Params']['Name'] = tmp_data['Params']['Name'] + '_0'
                split_data.append(tmp_data)
                
    return split_data


def expolartion_write(input_json):
    
    with open(input_json, 'r', encoding='utf-8') as fw:
        data = json.load(fw)
        
    work_dir = os.path.join(data['work_dir'],'exploration')
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    split_data = split_dict(data)
    
    for sp_d in split_data:
        path = os.path.join(work_dir,sp_d['Params']['Name'])
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            os.system('rm -rf  ' + path)
        dump = json.dumps(sp_d, indent=1)
        # print(dump)
        with open(os.path.join(path,'input.json'),'w') as f:
            f.write(dump)
        write_python(path)
        os.system('cp sub_exploration '+path)
        

def write_python(path):
    python_name = os.path.join(path,'input.py')
    if os.path.exists(python_name):
        os.remove(python_name)
    
    with open(python_name,'w') as f:
        f.write('''import json
from active_learning_expolartion import *
with open('input.json') as f:
    jdata = json.load(f)
exploration_one(jdata)
''')
    

        
