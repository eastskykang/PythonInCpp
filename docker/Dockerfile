FROM tensorflow/tensorflow:nightly-devel-gpu-py3

WORKDIR /tensorflow

# bazel upgrade
#RUN BAZEL_VERSION="0.15.0" && \
#    wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
#    chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
#    ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
#    rm bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

RUN git fetch --tags && git checkout v1.12.0

RUN ./configure && \
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package && \
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp