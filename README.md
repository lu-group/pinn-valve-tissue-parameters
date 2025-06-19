> Under Construction

# A Noninvasive Method for Determining Elastic Parameters of Valve Tissue Using Physics-Informed Neural Networks

The data and code for the paper [W. Wu, M. Daneker, C. Herz, H. Dewey, J.A. Weiss, A.M. Pouch, L. Lu & M.A. Jolley. A Noninvasive Method for Determining Elastic Properties of Valve Tissue Using Physics-Informed Neural Networks, *Acta Biomaterialia*, 200, 283-298, 2025](https://www.sciencedirect.com/science/article/abs/pii/S1742706125003472).

## Data
All data are in the folder [data](data). The name preceding ".npy" indicates the data for a specified example. For example, "HLHS_TV_data.npy" contains data for the HLHS tricuspid valve example.

## Code

All code are in the folder [src](src). The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde) v1.12.1. 

- [2D hollow cylinder](src/example1_hollow_cylinder.py)
- [2D deflected circular plate](src/example2_deflected_circular_plate.py)
- [3D cone](src/eexample3_cone.py)
- [3D HLHS tricuspid valve with Neo-Hookean material model](src/example4_HLHS_TV_NeoHookean.py)
- [3D HLHS tricuspid valve with Lee-Sacks material model](src/example4_HLHS_TV_LeeSacks.py)

To run the code:
```bash
python example1_hollow_cylinder.py
```

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{wu2025noninvasive,
  author  = {Wu, Wensi and Daneker, Mitchell and Herz, Christian and Dewey, Hannah and Weiss, Jeffrey A. and Pouch, Alison M. and Lu, Lu and Jolley, Matthew A.},
  title   = {A Noninvasive Method for Determining Elastic Parameters of Valve Tissue Using Physics-Informed Neural Networks}, 
  journal = {Acta Biomaterialia},
  volume  = {200},
  number  = {},
  pages   = {283-298},
  year    = {2025},
  doi     = {https://doi.org/10.1016/j.actbio.2025.05.021}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
