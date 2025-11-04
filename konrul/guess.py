import os
import subprocess
import torch
import sentencepiece as spm
from fairseq.models.transformer import TransformerModel

from konrul.constants import MODEL_DIR
from konrul.einsum_program import EinsumProgram

_real_load = torch.load

def _patched_load(f, *args, **kwargs):
  kwargs["weights_only"] = False
  return _real_load(f, *args, **kwargs)

torch.load = _patched_load


def load_model():
  model_dir = MODEL_DIR
  sp = spm.SentencePieceProcessor(model_file=str(os.path.join(model_dir,"spm.bpe.model")))

  model = TransformerModel.from_pretrained(
      model_name_or_path=str(model_dir),
      checkpoint_file="checkpoint_best.pt",
      data_name_or_path=str(model_dir),
      bpe=None
  )
  
  return sp, model


def remove_linebreak(benchmark, benchmark_name):
  output_file = f"/tmp/{benchmark_name}"
  with open(output_file, "w") as out:
    subprocess.run(
      ["tr", "-d", "\n"],
      stdin = open(benchmark, "r"),
      stdout = out,
      check=True
    )


def guess(benchmark):
  benchmark_name = os.path.basename(benchmark)
  remove_linebreak(benchmark, benchmark_name)
  sentence = ""
  with open(f"/tmp/{benchmark_name}") as f:
    sentence = f.read()

  sp, guesser = load_model()
  encoded = sp.encode(sentence, out_type = str)
  joined = " ".join(encoded)
  guess = guesser.translate(joined)
  decoded = sp.decode(guess.split())
  decoded = decoded.replace("\"", "")
  print("→", decoded)
  return EinsumProgram.from_string(decoded)




