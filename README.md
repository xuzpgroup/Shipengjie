This work is related to the following studies:  
[1] Shi P, Xu Z\*. Edge perfection of two-dimensional materials. *MRS Bulletin*, 2025, 50(5): 549-556.  
[2] Shi P, Xu Z\*. Strength of 2D glasses explored by machine-learning force fields. *Journal of Applied Physics*, 2024, 136(6).  
[3] Shi P, Xu Z. Exploring fracture of H-BN and graphene by neural network force fields. *Journal of Physics: Condensed Matter*, 2024, 36(41): 415401.  
[4] Shi P, Feng S, Xu Z\*. Non-equilibrium nature of fracture determines the crack paths. *Extreme Mechanics Letters*, 2024, 68: 102151.  

The relevant computational files for the above articles are organized as follows:  

- **`active_learning`**:  
  Contains the active learning framework for training machine-learning force fields (References [3] and [4]).  

- **`graphene`**:  
  - `pySIF.py`: Source code for calculating stress intensity factors.  
  - `DP_multilayer.py`: ASE force field file for simulating bilayer graphene (Reference [3]), supports Lennard-Jones potential and DeepPotential force fields.  
  - **`quasi_static_run`**: Example scripts for quasi-static calculations (Reference [4]).  
  - **`structures`**: Structural files used in Reference [4].  
  - **`KC_bilayer`**: KC bilayer graphene structure files (Reference [3]).  

- **`hbn/structures`**:  
  H-BN structure files used in Reference [3].  

- **`sio2`**:  
  - **`para-structure`**: Paracrystalline structures & input scripts (Reference [2]).  
  - **`np-structure`**: Nanopolycrystalline structures & input scripts (Reference [2]).  
  - **`crn-structure`**: Continuous random network structures & input scripts (Reference [2]).  
  - **`ncg-structure`**: Nanocrystalline glass structures & input scripts (Reference [2]).  
  - **`large_amorphous_structures`**: Large-scale amorphous structure files (References [1] and [2]).  
  - **`structures`**: Crystalline structure files for SiO₂ (Reference [1]).