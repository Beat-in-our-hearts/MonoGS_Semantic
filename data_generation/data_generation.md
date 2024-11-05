## Replica Data Generation

### 1.Download Replica Dataset
Download 3D models and info files from [Replica](https://github.com/facebookresearch/Replica-Dataset)

```
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
./download.sh ./replica_v1
```

This will download 17 zip files, totaling `34GB` of data.

### 2.Copy the `Replica_empty`
The `traj.txt` file stores random camera poses. Please place it in the `dataset` directory and rename it to `replica`.
```
cp -r Replica_empty/ /path/to/dataset/replica
```

### 3.Rendering 2D Images

####  Install Habitat-Sim 0.2.1
We recommend to use conda to install habitat-sim 0.2.1. Please build new environment.
```angular2html
conda create -n habitat_data_generation python=3.8.12 cmake=3.14.0 
conda activate habitat_data_generation
conda install habitat-sim=0.2.1 withbullet -c conda-forge -c aihabitat 
conda install numba=0.54.1
pip install pyyaml imgviz opencv-python plyfile trimesh==3.10.7
```

#### Run rendering with configs
Change path in configs.
```
python habitat_renderer.py --config replica_render_config_vMAP.yaml 
```

### 4.Options
#### Change camera intrinsic
We put the camera intrinsic in `Replica_empty/cam_params.json`.

In config file, `width: 1200 hfov:90(deg)` means `"camera": {"w": 1200,"h": 680,"fx": 600.0,"fy": 600.0,"cx": 599.5,"cy": 339.5}`

#### Render hdr image
If you want to render HDR images, check the `lighting: True` in config or run below command with args `--lighting`. After run **agian**, you will get `rgb_hdr` folder. Delete orignal rgb folder and rename `rgb_hdr` to `rgb`.
```
python habitat_renderer.py --config replica_render_config_vMAP.yaml --lighting
```
#### Change depth scale
The default depth sensor scale is `1000`. If you want to get the vis_depth image, rub below command with args `--depth_scale` `--depth_only`. Please change `scale` in `Replica_empty/cam_params.json`.

```
python habitat_renderer.py --config replica_render_config_vMAP.yaml --depth_scale 6553.5 --depth_only
```