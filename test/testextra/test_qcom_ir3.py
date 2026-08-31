# tests for the ir3 disassembly parser in test/mockgpu/qcom/ir3.py, these need no device
import unittest
from test.mockgpu.qcom.ir3 import Label, parse

class TestIR3Parse(unittest.TestCase):
  def test_register_flat_index(self):
    # a register is written <num>.<component> but indexes a flat file, so r2.x is register 8
    for text, idx in [("r0.x", 0), ("r0.w", 3), ("r1.x", 4), ("r2.z", 10), ("r3.y", 13)]:
      ins = parse(f"add.u {text}, r0.x, r0.x")
      self.assertEqual(ins.dst, ("r", idx, False, False, False), text)

  def test_half_and_const_files(self):
    ins = parse("cmps.u.lt hr0.z, r3.y, c1.x")
    self.assertEqual(ins.dst, ("r", 2, True, False, False))     # hr0.z: half register, flat index 2
    self.assertEqual(ins.srcs[0], ("r", 13, False, False, False))
    self.assertEqual(ins.srcs[1], ("c", 4, False, False, False))  # c1.x lives in the const file at flat index 4

  def test_repeat_and_delay_flags(self):
    ins = parse("(rpt2)(ss)(sy)add.f r1.x, r2.y, c3.z")
    self.assertEqual((ins.op, ins.rpt), ("add.f", 2))
    self.assertEqual(ins.flags, {"ss", "sy"})
    ins = parse("(nop3) ashr.b r0.w, r0.x, 31")
    self.assertEqual((ins.op, ins.nop, ins.rpt), ("ashr.b", 3, 0))

  def test_immediates(self):
    self.assertEqual(parse("shl.b r0.w, r0.w, 2").srcs[1], ("imm", 2))
    self.assertEqual(parse("add.u r0.x, r0.x, -5").srcs[1], ("imm", -5))
    self.assertEqual(parse("mov.u32u32 r0.x, 0x1f").srcs[0], ("imm", 31))

  def test_source_modifiers(self):
    ins = parse("add.f r0.x, -r1.y, |r2.z|")
    self.assertEqual(ins.srcs[0], ("r", 5, False, True, False))   # negated
    self.assertEqual(ins.srcs[1], ("r", 10, False, False, True))  # absolute value

  def test_memory_operands(self):
    ins = parse("ldg.u32 r2.w, g[r1.x], 1")
    self.assertEqual(ins.srcs[0], ("mem", "g", ("r", 4, False, False, False), 0))
    ins = parse("stg.u32 g[r1.x+4], r2.w, 1")
    self.assertEqual(ins.dst, ("mem", "g", ("r", 4, False, False, False), 4))

  def test_multi_source(self):
    ins = parse("mad.f32 r0.x, r1.y, r2.z, r3.w")
    self.assertEqual(len(ins.srcs), 3)
    self.assertEqual([s[1] for s in ins.srcs], [5, 10, 15])

  def test_no_operands(self):
    self.assertEqual(parse("nop").op, "nop")
    self.assertEqual(parse("nop").srcs, [])
    self.assertIsNone(parse("   "))

  def test_labels_and_branch_targets(self):
    # a label is its own line, and a branch names it with a leading #
    self.assertIsInstance(parse("l25:"), Label)
    self.assertEqual(parse("l25:").name, "l25")
    ins = parse("br p0.y, #l25")
    self.assertEqual(ins.op, "br")
    self.assertEqual(ins.srcs[0], ("p", 1, False, False, False))  # p0.y, the predicate file indexes flat too
    self.assertEqual(ins.srcs[1], ("label", "l25"))
    self.assertEqual(parse("jump #l6").srcs[0], ("label", "l6"))

  def test_unknown_operand_is_kept_raw(self):
    # anything the parser does not recognise stays visible instead of silently decoding wrong
    self.assertEqual(parse("sam (f32)(xyzw)r0.x, r1.y, s#0, t#0").srcs[-1], ("raw", "t#0"))

if __name__ == "__main__":
  unittest.main()
