.PHONY: audit build down smoke status test verify

audit:
	./scripts/public-audit.sh

build:
	./scripts/build-image.sh

status:
	./scripts/status.sh

smoke:
	./scripts/smoke-t2va.sh

verify:
	./scripts/verify-output.sh output/smoke-t2va-2x.mp4

test:
	docker run --rm --network none --entrypoint python \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-v "$(CURDIR):/workspace:ro" -w /workspace \
		minimax-h3-2x-dgx-spark:experimental \
		-m pytest -q -p no:cacheprovider tests

down:
	./scripts/stop-two-sparks.sh
