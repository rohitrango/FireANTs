.PHONY: docs tests tests-all build_fused_ops changelog copyright

docs:
	mkdocs serve -f docs/mkdocs.yml

tests:
	bash run_tests.sh

tests-all:
	bash run_tests.sh --all

build_fused_ops:
	cd fused_ops && python setup.py build_ext && python setup.py install && cd ..

changelog:
	python scripts/update_versions_and_changelog.py

copyright:
	python autocopyright.py
