import numpy as np
import os
from ase.io import read,write
from deepmd.calculator import DP
height = 100.
ev2J= 1.6021766208e-19/1e-20

start_strain = 1.09
delta_strain = 0.04
step = 4

os.system('mpiexec.hydra -n 32 lmp -in in.graphene')
atom = read('dump.atom',format='lammps-dump-text',specorder=['C'])
pot = DP('../frozen_model.pb')
atom.set_calculator(pot)
stress = atom.get_stress()
energy = atom.get_potential_energy()

with open('qusi_tension.log','a') as f:
    f.write('strain\t\tenergy\t\tpxx\t\tpyy\n')
    f.write('%.5f\t\t%.5f\t%.5f\t%5f\n'%(0.0,energy,stress[0]*ev2J*height,stress[1]*ev2J*height))

allatom = []
allatom.append(atom.copy())

cell = np.array(atom.get_cell())
cell[:,1] *= start_strain
atom.set_cell(cell,scale_atoms=True)
write('data_tension.lmp',atom,format='lammps-data')

os.system('mpiexec.hydra -n 32 lmp -in in.tension_large')

dstrain = delta_strain/step
for i in range(1,step+1):
    atom = read('dump_tension.atom',format='lammps-dump-text',specorder=['C'])
    atom.set_calculator(pot)
    stress = atom.get_stress()
    energy = atom.get_potential_energy()
    with open('qusi_tension.log','a') as f:
        f.write('%.5f\t\t%.5f\t%.5f\t%5f\n'%(start_strain+(i-1)*dstrain,energy,stress[0]*ev2J*height,stress[1]*ev2J*height))
    allatom.append(atom.copy())
    cell = np.array(atom.get_cell())
    cell[:,1] *= (start_strain+i*dstrain)/(start_strain+(i-1)*dstrain)
    atom.set_cell(cell,scale_atoms=True)
    write('data_tension.lmp',atom,format='lammps-data')
    os.system('mpiexec.hydra -n 32 lmp -in in.tension')

atom = read('dump_tension.atom',format='lammps-dump-text',specorder=['C'])
atom.set_calculator(pot)
stress = atom.get_stress()
energy = atom.get_potential_energy()
with open('qusi_tension.log','a') as f:
    f.write('%.5f\t\t%.5f\t%.5f\t%5f\n'%(start_strain+(i)*dstrain,energy,stress[0]*ev2J*height,stress[1]*ev2J*height))
allatom.append(atom.copy())
write('gra.xyz',allatom,format='extxyz')

