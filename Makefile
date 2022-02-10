docker.build:
	docker build . -t spline_contour

docker.test:
	docker run spline_contour conda run pytest --collect-only --nbmake "data.ipynb" "training.ipynb" "prediction.ipynb"

repo2docker.build:
	repo2docker --no-run --image-name spline_contour .

# repo2docker.tests:

mamba.build.env:
	mamba env create -f environment.yml