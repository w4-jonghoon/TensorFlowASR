FROM tensorflow/tensorflow:2.5.0

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev git build-essential cmake screen

# Clear cache
RUN apt clean && apt-get clean

# Install dependencies
COPY requirements.txt /
RUN pip --no-cache-dir install -r /requirements.txt

# Install rnnt_loss
COPY scripts /scripts
ARG install_rnnt_loss=false
ARG using_gpu=false
RUN if [ "$install_rnnt_loss" = "true" ] ; \
    then if [ "$using_gpu" = "true" ] ; then export CUDA_HOME=/usr/local/cuda ; else echo 'Using CPU' ; fi \
    && ./scripts/install_rnnt_loss.sh \
    else echo 'Using pure TensorFlow'; fi

COPY examples /examples
COPY tensorflow_asr /tensorflow_asr
COPY vocabularies /vocabularies
ENV PYTHONPATH /:$PYTHONPATH
