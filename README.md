# KONRUL: Using Lowering to Lift Tensor Code

KONRUL is a code lifter that correctly lifts existing tensor algebra C code to einsum notation,
the basis of tensor contraction DSLs. 

KONRUL is based on the _Guess, Measure & Edit_ framework [Magalhães et. al. PACT'25]. It uses a Transformer model to guess a solution, compiler lowering to infer a specification for the guess, program
similarity metrics to measure the difference between the specifications and guide a search of edit rules, and finally testing and model checking to verify the solution. KONRUL's guesser model is also availabe on [HuggingFace](https://huggingface.co/jwesleysm/konrul-guesser ).

Full details of KONRUL are available in the reference paper. 

### Target language 
We use extended einsum notation as
our target language. Einstein summation (einsum) notation
is a high-level language to express tensor contractions.
An einsum program contains an indexing term for each tensor
where the indices shared between terms are multiplied, and
the indices that are not shared with the left-hand side are
implicitly summed. Original einsum notation does not support
all operations used in tensor algebra, so DSLs adopt an
extended version. Once we have a valid solution, KONRUL exports it to PyTOrch

## Installation

Currently, KONRUL is provided as a [Docker](https://www.docker.com/) artifact. You can install Docker following the official [instructions](https://docs.docker.com/get-docker/). To build the KONRUL Docker image, run the command below within this repo directory directory:

_Important_: Depending on privilegies, you might have to run Docker as sudo or have your user added to a group that can run Docker without root access.

```
$ docker build -t konrul-docker -f docker/Dockerfile .
```

Once the image is built, that can be executed with the command:

```
$ docker run -ti -v /root/konrul konrul-docker
```

This will run KONRUL Docker image and leave you inside a container with all KONRUL completely configured. Notice that, by default, KONRUL Dockerfile downloads LLVM for x86. This can be changed in case you want to run it on other architectures.

## Usage

KONRUL takes as input the original program in C, the MLIR version of it, the set of tests (I/O samples) and the number of attempts to run the whole lifting pipeline.

```
$ python3 main.py <path-to-c-program> <path-to-mlir-program> <path-to-IO-samples> <number-of-pipeline-iterations>
```
## Citation

If you use KONRUL, please refer the reference paper:

```bibtex
@inproceedings{Magalhaes2025GuessMeasureEdit,
  author       = {José Wesley De Souza Magalhães and Jackson Woodruff and Jordi Armengol-Estapé and Alexander Brauckmann and Luc Jaulmes and Elizabeth Polgreen and Michael O'Boyle},
  title        = {Guess, Measure \& Edit: Using Lowering to Lift Tensor Code},
  booktitle    = {Proceedings of the 2025 ACM/IEEE International Conference on Parallel Architectures and Compilation Techniques (PACT 2025)},
  year         = {2025},
}
```


## Notes
  - We provide some benchmarks that are evaluated in the paper, but not all. We're checking licensing matters and will release the full benchmark suite soon
  - We will soon include instructions to generate your own test cases for a new program
  - We will include the link to the full paper once the conference proceedings are available
