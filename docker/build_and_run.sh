sudo docker build -t cuda11.8-opengl-ubuntu22.04 --network host .

sudo xhost + 
export DISPLAY=:1
sudo docker run -it --name=gsdff_slam_cudagl_11.8  --network host \
	--gpus '"device=2,3"' -v $DATA_DISK:/home/data \
	-v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/dri:/dev/dri \
    --device=/dev/dri --group-add video \
    --env=DISPLAY=$DISPLAY --env=QT_X11_NO_MITSHM=1 \
    --env=__NV_PRIME_RENDER_OFFLOAD=1 \
    --env=__GLX_VENDOR_LIBRARY_NAME=nvidia \
    --env=__NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0 \
    cuda11.8-opengl-ubuntu22.04:latest /bin/bash