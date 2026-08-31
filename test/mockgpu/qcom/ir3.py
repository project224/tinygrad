import ctypes, re, tempfile
from tinygrad.runtime.autogen import mesa, libc

# ***** disassembly *****
# mesa owns the ir3 encoding, so rather than decode the bit fields by hand (the isaspec reuses field names across
# operands, which is easy to get subtly wrong) we let mesa format each instruction and parse its text.

def disasm(lib:bytes, gpu_id:int=630) -> list[str]:
  """Returns one line of mesa's ir3 disassembly per instruction, without its address prefix."""
  with tempfile.TemporaryFile('w+') as tf:
    fp = libc.fdopen(tf.fileno(), b"w")
    mesa.ir3_isa_disasm(lib, len(lib), ctypes.cast(fp, ctypes.POINTER(mesa.struct__IO_FILE)),
                        mesa.struct_isa_decode_options(gpu_id, True, 0, True))
    libc.fflush(fp)
    tf.seek(0)
    text = tf.read()
  return [l.strip() for l in text.split("\n") if l.strip()]

# ***** parsing *****

FLAGS = ("sy", "ss", "jp", "ul", "eq")  # sync and scheduling flags, none of which affect results in an emulator
# the p file holds the predicates cmps writes and br reads, and indexes flat like r and c do
REG = re.compile(r"^(?P<neg>-)?(?P<abs>\|)?(?P<h>h)?(?P<file>r|c|p)(?P<num>\d+)\.(?P<comp>[xyzw])\|?$")
LABEL = re.compile(r"^(l\d+):$")            # a label sits alone on its own line
LABEL_REF = re.compile(r"^#(l\d+)$")        # branches name their target as "#l25"
# control flow and synchronisation write no register, so their first operand is a source and not a destination
NO_DST = ("br", "brao", "braa", "bany", "ball", "jump", "call", "ret", "kill", "end", "nop", "bar", "fence", "predt", "predf", "prede")
# a memory operand names its address space and an address, eg "g[r1.x]" or "l[r2.y+4]"
MEM = re.compile(r"^(?P<space>[a-z]+)\[(?P<addr>[^\]]+)\]$")

class Label:
  """A branch target. Labels are their own disassembly line, so they carry no instruction."""
  __slots__ = ("name",)
  def __init__(self, name:str): self.name = name
  def __repr__(self): return f"{self.name}:"

class Inst:
  __slots__ = ("op", "dst", "srcs", "rpt", "nop", "flags", "line")
  def __init__(self, op:str, dst, srcs:list, rpt:int, nop:int, flags:set, line:str):
    self.op, self.dst, self.srcs, self.rpt, self.nop, self.flags, self.line = op, dst, srcs, rpt, nop, flags, line
  def __repr__(self): return self.line

def _operand(tok:str):
  """A register is (file, flat index, half); an immediate is ('imm', value); anything else stays a raw string."""
  tok = tok.strip()
  if (m := REG.match(tok)) is not None:
    idx = int(m.group("num")) * 4 + "xyzw".index(m.group("comp"))
    return (m.group("file"), idx, m.group("h") == "h", m.group("neg") == "-", m.group("abs") == "|")
  if re.fullmatch(r"[-+]?\d+", tok): return ("imm", int(tok))
  if re.fullmatch(r"[-+]?(0[xX])?[0-9a-fA-F]+", tok) and tok.lower().startswith("0x"): return ("imm", int(tok, 16))
  if re.fullmatch(r"[-+]?\d*\.\d+", tok): return ("immf", float(tok))
  if (m := LABEL_REF.match(tok)) is not None: return ("label", m.group(1))
  if (m := MEM.match(tok)) is not None:
    base, _, off = m.group("addr").partition("+")
    return ("mem", m.group("space"), _operand(base), int(off) if off.strip() else 0)
  return ("raw", tok)

def parse(line:str) -> Inst|Label|None:
  """Parses one disassembly line, eg '(nop3) add.u r2.x, c0.w, r0.w' or '(rpt2)(ss)cov.u32f32 r1.x, r0.x'."""
  if not (line := line.strip()): return None
  if (m := LABEL.match(line)) is not None: return Label(m.group(1))
  rpt, nop, flags = 0, 0, set()
  while (m := re.match(r"^\((\w+?)(\d*)\)\s*", line)):
    name, num = m.group(1), m.group(2)
    if name == "rpt": rpt = int(num)
    elif name == "nop": nop = int(num)
    elif name in FLAGS: flags.add(name)
    line = line[m.end():]
  if not line: return None
  op, _, rest = line.partition(" ")
  ops = [_operand(x) for x in rest.split(",")] if rest.strip() else []
  if op.split(".")[0] in NO_DST: return Inst(op, None, ops, rpt, nop, flags, line)
  return Inst(op, ops[0] if ops else None, ops[1:], rpt, nop, flags, line)

def decode(lib:bytes, gpu_id:int=630) -> tuple[list[Inst], dict[str, int]]:
  """The instruction stream with labels lifted out, plus the index each label points at."""
  insts:list[Inst] = []
  labels:dict[str, int] = {}
  for l in disasm(lib, gpu_id):
    if (i := parse(l)) is None: continue
    if isinstance(i, Label): labels[i.name] = len(insts)
    else:
      insts.append(i)
      if i.op == "end": break  # the shader is padded with nops out to the end of the page
  return insts, labels
