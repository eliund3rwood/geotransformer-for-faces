help:
	@echo "help             	-- print this help"
	@echo "build            	-- builds the docker image"
	@echo "dockershell      	-- open terminal in docker container from image"

build-image:
	docker build -t geotransformer --build-arg UID=`id -u` --build-arg GID=`id -g` -f ./docker/dockerfile .

dockershell:
	docker run -it --rm --runtime nvidia \
            --gpus all \
            -h geotransformer \
	    	-v "$(PWD)":/workspace/src \
			-v /data:/data \
	    	-w=/workspace/src \
			--name geotransformer-shell \
			--shm-size=64g \
			-p 8890:8890 \
			geotransformer

train-faces: 
	docker run --rm -d --runtime nvidia \
            --gpus all \
            -h geotransformer \
			-e CUDA_VISIBLE_DEVICES=0 \
	    	-v "$(PWD)":/workspace/src \
			-v /data:/data \
	    	-w=/workspace/src/experiments/geotransformer.faces.stage4.gse.k3.max.oacl.stage2.sinkhorn \
			--name geotransformer-train \
			--shm-size=64g \
			geotransformer python trainval.py

train-faces-downsampled: 
	docker run --rm -d --runtime nvidia \
            --gpus all \
            -h geotransformer \
			-e CUDA_VISIBLE_DEVICES=0 \
	    	-v "$(PWD)":/workspace/src \
			-v /data:/data \
	    	-w=/workspace/src/experiments/geotransformer.faces.stage4.gse.k3.max.oacl.stage2.sinkhorn \
			--name geotransformer-train \
			--shm-size=64g \
			geotransformer python trainval_downsampled.py
