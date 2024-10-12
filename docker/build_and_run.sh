sudo docker build -t cuda11.8-opengl-ubuntu22.04 --network host .
sudo xhost +
sudo docker run -it --name=gs_semantic_slam_cudagl_11.8  --network host \
	--gpus all -v $HOME/docker_temp/gs_slam_temp_data:/home/data \
	-v /tmp/.X11-unix:/tmp/.X11-unix --device=/dev/dri --group-add video --env=DISPLAY=:0 --env=QT_X11_NO_MITSHM=1 \
  	cuda11.8-opengl-ubuntu22.04:latest /bin/bash