train:
	python train.py --data shipcam.yaml --cfg yolov5s.yaml --weights '' --batch-size 16 --epochs 1

trainBW:
	python train.py --data shipcam.yaml --cfg yolov5s.yaml --weights '' --batch-size 16 --epochs 1 --oceanBW
