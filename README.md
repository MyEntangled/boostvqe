# Boost VQEs with DBI

<img src="figures/diagram.png" style="solid #000; max-width:600px; max-height:1000px;">
Boosting variational eigenstate preparation algorithms limited by training and not device coherence by diagonalization double-bracket iteration.

## Installation instructions

The package can be installed by source after cloning the repository:

```sh
pip install -e .
```

will install `boostvqe 0.0.1` and activate a dedicated working shell.

## Code structure

The file `main.py` performs boosted VQE training.

The source code is located in `./src/boostvqe/.` and its composed of:

* `ansatze.py`: contains circuit used by VQE
* `utils.py`: contains utils function used by `main.py`
* `plotscripts.py`: plotting functions.
* `compiling_XXZ.py`: compilation for XXZ model.

## Example

It follows a python snippet explaining how to run the boosting

```py

from boostvqe.boost import main
from boostvqe.ansatze import hdw_efficient

from qibo.models.dbi.double_bracket import DoubleBracketGeneratorType

help(main) #If you need some info

circuit = hdw_efficient(nqubits=2, nlayers=2)

main(circuit, "test_output", mode = DoubleBracketGeneratorType.group_commutator)
```

# Tutorials

Some useful notebooks to understand how the library works, are collected [here](notebooks/notebooks_links.md).

# Reference and citation

For more details about this project and citations, please refer to [the article](https://www.arxiv.org/abs/2408.03987).

<img src="figures/hw_preserving_XXZ_10Q3L42S_cma_jumps.png" style="solid #000; max-width:600px; max-height:1000px;">
