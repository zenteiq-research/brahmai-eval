# brahmai-eval runtime image, built entirely from public images and PyPI (nothing proprietary is baked in).
#
#   docker build -t brahmai-eval .
#   docker run --rm --gpus all --ipc=host -v $PWD/results:/work/results -v $HOME/.cache:/root/.cache \
#       brahmai-eval brahmai-eval run --model Qwen/Qwen2.5-1.5B-Instruct --suite code,math
#
# The base image ships vLLM 0.25.1 with its matching torch/CUDA; the rest is pure-Python.
FROM vllm/vllm-openai:v0.25.1

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 HF_HUB_ENABLE_HF_TRANSFER=0
WORKDIR /work
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY brahmai_eval ./brahmai_eval
COPY tasks ./tasks
RUN pip install --no-deps -e . \
 && pip install "lm-eval[ifeval,math]==0.4.12" "huggingface_hub>=0.30" "datasets>=3.0" pyyaml "sacrebleu>=2.4" "rouge-score>=0.1.2" \
      "evaluate>=0.4" "nltk>=3.9" "langdetect>=1.0.9" "immutabledict>=4" "antlr4-python3-runtime==4.11" "pyarrow>=15" requests \
 && python -c "import vllm, lm_eval, torch; print('vllm', vllm.__version__, '| lm_eval', lm_eval.__version__, '| torch', torch.__version__)"
ENTRYPOINT []
CMD ["brahmai-eval", "--help"]
