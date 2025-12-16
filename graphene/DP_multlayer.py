import numpy as np
from ase.neighborlist import NeighborList
from ase.stress import full_3x3_to_voigt_6_stress
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from ase.calculators.calculator import (
    Calculator, all_changes, PropertyNotImplementedError, all_changes)
from deepmd import DeepPotential
if TYPE_CHECKING:
    from ase import Atoms
    
    
    
class DP_multilayer(Calculator):
    implemented_properties = ['energy','free_energy', 'forces','stress']
    default_parameters = {
        'epsilon': 1.0,
        'sigma': 1.0,
        'rc': None,
        'ro': None,
        'smooth': False,
    }
    def __init__(self,
                 model: Union[str, "Path"],
                 label: str = "DP",
                 type_dict: Dict[str, int] = None,
                 sigma = None,
                 epsilon = None,
                 rc = None,
                 ro = None,
                 smooth=False,
                 **kwargs
                 ):
        Calculator.__init__(self, **kwargs)
        self.sigma = sigma
        self.epsilon = epsilon
        self.rc = rc
        self.ro = ro
        self.nl = None
        self.smooth = smooth
        if self.rc is None:
            self.rc = 3 * self.sigma
        if self.ro is None:
            self.ro = 0.66 * self.rc
        
        
        self.dp = DeepPotential(str(Path(model).resolve()))
        if type_dict:
            self.type_dict = type_dict
        else:
            self.type_dict = dict(
                zip(self.dp.get_type_map(), range(self.dp.get_ntypes()))
            )
    def calculate(
        self,
        atoms=None,
        properties=['energy','free_energy', 'forces','stress'],
        system_changes=all_changes,
    ):
        
        properties = self.implemented_properties
        if atoms is not None:
            self.atoms = atoms.copy()
        
        # LJ part
        natoms = len(self.atoms)
        sigma = self.sigma
        epsilon = self.epsilon
        rc = self.rc
        ro = self.ro
        smooth = self.smooth

        tags = self.atoms.get_tags()
        
        if self.nl is None or 'numbers' in system_changes:
            self.nl = NeighborList(
                [rc / 2] * natoms, self_interaction=False, bothways=True
            )
        self.nl.update(self.atoms)
        positions = self.atoms.get_positions()
        cell = self.atoms.get_cell()
        
         # potential value at rc
        e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
        
        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))
        
        for ii in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(ii)
            
            I = np.argwhere(self.atoms[ii].tag != tags[neighbors]).flatten()
            
            cells = np.dot(offsets, cell)
            distance_vectors = (positions[neighbors] + cells - positions[ii])[I]
            
            r2 = (distance_vectors ** 2).sum(1)
            c6 = (sigma ** 2 / r2) ** 3
            c6[r2 > rc ** 2] = 0.0
            c12 = c6 ** 2
            
            if smooth:
                cutoff_fn = cutoff_function(r2, rc**2, ro**2)
                d_cutoff_fn = d_cutoff_function(r2, rc**2, ro**2)
            
            pairwise_energies = 4 * epsilon * (c12 - c6)
            pairwise_forces = -24 * epsilon * (2 * c12 - c6) / r2  # du_ij
            
            if smooth:
                # order matters, otherwise the pairwise energy is already modified
                pairwise_forces = (
                    cutoff_fn * pairwise_forces + 2 * d_cutoff_fn * pairwise_energies
                )
                pairwise_energies *= cutoff_fn
            else:
                pairwise_energies -= e0 * (c6 != 0.0)
            
            pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors
            
            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies
            forces[ii] += pairwise_forces.sum(axis=0)

            stresses[ii] += 0.5 * np.dot(
                pairwise_forces.T, distance_vectors
            )  # equivalent to outer product
            
        stress = 0
        # no lattice, no stress
        if self.atoms.cell.rank == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            stress += stresses.sum(axis=0) / self.atoms.get_volume()
        
        energy = energies.sum()
        
        
        
        # DP part
        num_tags = sorted(list(set(tags)))
        for ii in num_tags:
            I = np.argwhere(tags==ii).flatten()
            new_atom = self.atoms[I].copy()
            
            coord = new_atom.get_positions().reshape([1, -1])
            if sum(new_atom.get_pbc()) > 0:
                cell = new_atom.get_cell().reshape([1, -1])
            else:
                cell = None
            symbols = new_atom.get_chemical_symbols()
            atype = [self.type_dict[k] for k in symbols]
            e, f, v = self.dp.eval(coords=coord, cells=cell, atom_types=atype)
            energy +=  e[0][0]
            forces[I] += f[0]
            if "stress" in properties:
                if sum(new_atom.get_pbc()) > 0:
                    temp_stress = -0.5 * (v[0].copy() + v[0].copy().T) / new_atom.get_volume()
                    stress += temp_stress.flat[[0, 4, 8, 5, 2, 1]]
            else:
                raise PropertyNotImplementedError
                
        # Results
        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress
        
    
    
def cutoff_function(r, rc, ro):
    """Smooth cutoff function.

    Goes from 1 to 0 between ro and rc, ensuring
    that u(r) = lj(r) * cutoff_function(r) is C^1.

    Defined as 1 below ro, 0 above rc.

    Note that r, rc, ro are all expected to be squared,
    i.e. `r = r_ij^2`, etc.

    Taken from https://github.com/google/jax-md.

    """

    return np.where(
        r < ro,
        1.0,
        np.where(r < rc, (rc - r) ** 2 * (rc + 2 * r - 3 * ro) / (rc - ro) ** 3, 0.0),
    )


def d_cutoff_function(r, rc, ro):
    """Derivative of smooth cutoff function wrt r.

    Note that `r = r_ij^2`, so for the derivative wrt to `r_ij`,
    we need to multiply `2*r_ij`. This gives rise to the factor 2
    above, the `r_ij` is cancelled out by the remaining derivative
    `d r_ij / d d_ij`, i.e. going from scalar distance to distance vector.
    """

    return np.where(
        r < ro,
        0.0,
        np.where(r < rc, 6 * (rc - r) * (ro - r) / (rc - ro) ** 3, 0.0),
    )

        