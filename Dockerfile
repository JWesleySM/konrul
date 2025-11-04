############################ Image #############################

FROM ubuntu:20.04
ENV TZ="Europe/London"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
RUN echo $TZ > /etc/timezone && rm -rf /var/lib/apt/lists/*



############################ General dependencies ############################

RUN apt update -y
RUN apt install -y cmake wget python3 python3-pip python3-dev tar git python3-venv
RUN apt install -y ninja-build bison flex libtool jq lld libgmp3-dev build-essential libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libboost-all-dev default-jdk maven

############################ Clone and setup KONRUL ############################
RUN git clone https://github.com/JWesleySM/konrul
WORKDIR /konrul

RUN python3 -m venv .venv/venv
ENV PATH="/konrul/.venv/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install numpy==1.24.4 pybind11 toml pyparsing pymlir 

RUN pwd && ls && ./verification/build_tools/build_dependencies.sh && \
    ./verification/build_tools/build_mlirSynth.sh && \
    python -m pip install --upgrade pip && \
    python -m pip install -r deps/llvm-project/mlir/python/requirements.txt && \
    python -m pip install jax==0.4.25 jaxlib==0.4.25

RUN git clone https://github.com/tensor-compiler/taco && \
    cd taco && mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON=ON .. && \
    make -j"$(nproc)"

# Download and unpack LLVM 14
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
    tar -xf clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
    mv clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04 llvm && \
    rm clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz

# Persistent environment variables
ENV PATH="$PATH:/konrul/verification/deps/cvc5/build/bin:/konrul/verification/deps/cbmc/build/bin"
ENV PYTHONPATH="/konrul/taco/build/lib:/konrul/verification/build/python_packages/synth:$PYTHONPATH"

#ENV PATH="/konrul/.venv/ven/bin:$PATH"
#WORKDIR /konrul
ENTRYPOINT ["/bin/bash"]
