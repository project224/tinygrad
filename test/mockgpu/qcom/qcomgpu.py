from tinygrad.helpers import to_mv
from tinygrad.runtime.autogen import mesa

# A6XX command streams are PM4: type4 packets write a run of registers, type7 packets are opcodes with a payload.
def pkt_type(hdr:int) -> int: return hdr & 0xF0000000

class QCOMGPU:
  """Models enough of an Adreno A6XX to run a compute dispatch: a register file, the state a CP_LOAD_STATE6 sets up,
  and the handful of CP_ opcodes tinygrad's QCOMComputeQueue emits."""
  def __init__(self, gpu_id:int):
    self.gpu_id = gpu_id
    self.regs:dict[int, int] = {}
    self.state:dict[str, int] = {}   # base addresses set by CP_LOAD_STATE6_FRAG, keyed by state block
    self.timestamp = 0

  def reg64(self, reg:int) -> int: return self.regs.get(reg, 0) | (self.regs.get(reg + 1, 0) << 32)

  def execute(self, addr:int, dwords:int):
    q = to_mv(addr, dwords * 4).cast('I')
    i = 0
    while i < dwords:
      hdr = q[i]
      if pkt_type(hdr) == mesa.CP_TYPE4_PKT:
        reg, cnt = (hdr >> 8) & 0x3FFFF, hdr & 0x7F
        for j in range(cnt): self.regs[reg + j] = q[i + 1 + j]
        i += 1 + cnt
      elif pkt_type(hdr) == mesa.CP_TYPE7_PKT:
        opcode, cnt = (hdr >> 16) & 0x7F, hdr & 0x3FFF
        self._exec_pkt7(opcode, [q[i + 1 + j] for j in range(cnt)])
        i += 1 + cnt
      else: raise RuntimeError(f"unknown pm4 packet header {hdr:#x} at dword {i}")

  def _exec_pkt7(self, opcode:int, vals:list[int]):
    # everything runs synchronously, so idle waits and cache maintenance have nothing to do
    if opcode in (mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES, mesa.CP_SET_MARKER, mesa.CP_NOP): return
    if opcode == mesa.CP_WAIT_REG_MEM: return  # the value it waits on was already written by an earlier packet
    if opcode == mesa.CP_EVENT_WRITE:
      # CACHE_FLUSH_TS carries an address and a value to write once the flush retires; a bare event has no payload
      if len(vals) >= 4: to_mv(vals[1] | (vals[2] << 32), 4).cast('I')[0] = vals[3]
      return
    if opcode == mesa.CP_REG_TO_MEM:
      # only used for the always-on counter, which backs profiling timestamps
      self.timestamp += 1
      to_mv(vals[1] | (vals[2] << 32), 8).cast('Q')[0] = self.timestamp
      return
    if opcode == mesa.CP_LOAD_STATE6_FRAG:
      state_type = (vals[0] >> mesa.CP_LOAD_STATE6_0_STATE_TYPE__SHIFT) & 0x3
      state_block = (vals[0] >> mesa.CP_LOAD_STATE6_0_STATE_BLOCK__SHIFT) & 0xF
      self.state[f"{state_type}:{state_block}"] = vals[1] | (vals[2] << 32)
      return
    if opcode in (mesa.CP_EXEC_CS, mesa.CP_RUN_OPENCL): return self._dispatch(opcode, vals)
    raise RuntimeError(f"unsupported pm4 opcode {mesa.__dict__.get(f'CP_{opcode}', opcode)} ({opcode:#x})")

  def _dispatch(self, opcode:int, vals:list[int]):
    ndrange = mesa.REG_A6XX_SP_CS_NDRANGE_0
    nd0 = self.regs.get(ndrange, 0)
    local_size = tuple(((nd0 >> s) & 0x3FF) + 1 for s in (mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEX__SHIFT,
                                                          mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__SHIFT,
                                                          mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__SHIFT))
    if opcode == mesa.CP_EXEC_CS: grid = tuple(vals[1 + i] & 0xFFFF for i in range(3))
    else: grid = tuple(self.regs.get(ndrange + 7 + 2 * i, 0) for i in range(3))  # RUN_OPENCL takes the grid from the registers

    prog_addr = self.reg64(mesa.REG_A6XX_SP_CS_CNTL_0 + 4)
    prog_off = self.regs.get(mesa.REG_A6XX_SP_CS_CNTL_0 + 3, 0)
    kernargs = self.state.get(f"{mesa.ST_CONSTANTS}:{mesa.SB6_CS_SHADER}", 0)

    raise NotImplementedError(
      f"ir3 emulator not implemented yet: would dispatch shader at {prog_addr + prog_off:#x} "
      f"grid={grid} local={local_size} kernargs={kernargs:#x}")
