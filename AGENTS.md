All changes to .py files MUST be accompanied by full and explicit type annotations.

After every set of changes to a .py file, run `PYTHONPATH=. INPUT_ROOT="" SRC_ROOT="" pytest` to make sure everything passes. You absolutely must fix any failing tests. 

When asked to write tests, never modify any existing source code files in src/. Only existing test files may be modified. New test files may be created if necessary.

When writing a command line interface with argparse, always parse the arguments into a well-typed dataclass. The raw argparse object should never be returned by a function.

When running commands, run them directly in the provided shell and do not use another shell command like `zsh -lc`.
