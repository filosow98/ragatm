
PYMAIN = -m src index
PYCACHE = __pycache__ .mypy_cache .uv_cache src/__pycache__

.PHONY: install run debug clean fclean lint test devenv

all: install run

install:
	uv sync

run:
	uv run $(PYMAIN)

debug:
	uv run -m pdb $(PYMAIN)

clean:
	-rm -rf $(PYCACHE)

fclean: clean
	-rm -rf ./.venv
	-rm -rf ~/goinfre/huggingface
	-rm -rf ~/goinfre/RAGAtM/.venv
	-rm -rf ~/goinfre/RAGAtM/.uv_cache

lint:
	@-flake8 . 
	@-mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

test:
	uv run pytest ./src

devenv:
	-mkdir ~/goinfre/huggingface
	-mkdir ~/goinfre/RAGAtM
	-mkdir ~/goinfre/RAGAtM/.venv
	-mkdir ~/goinfre/RAGAtM/.uv_cache
	-ln -s ~/goinfre/huggingface ~/.cache/huggingface
	-ln -s ~/goinfre/RAGAtM/.venv ./.venv
	-ln -s ~/goinfre/RAGAtM/.uv_cache ./.uv_cache
	-uv venv --clear
	-uv sync
