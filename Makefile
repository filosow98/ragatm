
PYMAIN = src
PYCACHE = __pycache__ .mypy_cache .uv_cache src/__pycache__

.PHONY: install run debug clean fclean lint test devenv

all: install run

install:
	uv sync

run:
	uv run python -m $(PYMAIN)

debug:
	uv run python -m pdb -m $(PYMAIN)

clean:
	rm -rf $(PYCACHE)

fclean: clean
	rm -rf ~/goinfre/huggingface
	rm -rf ~/goinfre/CallMeMaybe/.venv
	rm -rf ~/goinfre/CallMeMaybe/.uv_cache

MYPY_LINT_EXCLUDE = --exclude ./llm_sdk --exclude ./.venv --exclude \
	 ./moulinette --exclude ./.uv_cache
FLAKE8_LINT_EXCLUDE = --exclude ./llm_sdk,./.venv,./moulinette,./.uv_cache

lint:
	@-flake8 . \
		$(FLAKE8_LINT_EXCLUDE)
	@-mypy . \
		$(MYPY_LINT_EXCLUDE) \
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
