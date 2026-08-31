import functools, mmap
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFileDesc, VirtFile
from test.mockgpu.qcom.qcomgpu import QCOMGPU

A630_CHIP_ID = 0x06030001  # ops_qcom.py reads the arch out of this as a<major><minor><patch>

class QCOMFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver

  def ioctl(self, fd, request, argp): return self.driver.ioctl(request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset): return self.driver.mmap(start, sz, prot, flags, offset)

class QCOMDriver(VirtDriver):
  """Mocks /dev/kgsl-3d0. Unlike NV and AMD there is no mmio doorbell or ring buffer: work is submitted with a single
  ioctl carrying a pm4 command buffer, so a submit just runs the command stream inline."""
  def __init__(self):
    super().__init__()
    self.tracked_files += [VirtFile('/dev/kgsl-3d0', functools.partial(QCOMFileDesc, driver=self))]
    self.gpu = QCOMGPU(A630_CHIP_ID)
    self.allocs:dict[int, tuple[int, int]] = {}  # id -> (host address, size), 0 until mmap'd
    self.next_alloc_id, self.next_ctx, self.timestamp, self._fd = 1, 1, 0, 1 << 30

  def open(self, name, flags, mode, virtfile): return virtfile.fdcls(self._alloc_fd())
  def _alloc_fd(self):
    self._fd += 1
    return self._fd

  def mmap(self, start, sz, prot, flags, offset):
    # kgsl maps an allocation by passing its id shifted into the mmap offset
    addr = libc.mmap(start, sz, prot, flags | mmap.MAP_ANONYMOUS, -1, 0)
    if addr == 0xffffffffffffffff: raise RuntimeError("mock kgsl mmap failed")
    if (aid := offset // 0x1000) in self.allocs: self.allocs[aid] = (addr, self.allocs[aid][1])
    return addr

  def ioctl(self, req, argp):
    nr = req & 0xff
    if nr == 0x02: return self._device_getproperty(argp)
    if nr == 0x13:  # DRAWCTXT_CREATE
      ctx = kgsl.struct_kgsl_drawctxt_create.from_address(argp)
      ctx.drawctxt_id, self.next_ctx = self.next_ctx, self.next_ctx + 1
      return 0
    if nr == 0x45:  # GPUOBJ_ALLOC
      alloc = kgsl.struct_kgsl_gpuobj_alloc.from_address(argp)
      alloc.id, self.next_alloc_id = self.next_alloc_id, self.next_alloc_id + 1
      alloc.mmapsize = alloc.size
      self.allocs[alloc.id] = (0, alloc.size)
      return 0
    if nr == 0x15:  # MAP_USER_MEM: gpu and cpu share an address space here, so mapping is the identity
      um = kgsl.struct_kgsl_map_user_mem.from_address(argp)
      um.gpuaddr = um.hostptr
      return 0
    if nr == 0x4A: return self._gpu_command(argp)
    if nr in (0x06, 0x07): return 0             # WAITTIMESTAMP: submits already completed inline
    if nr in (0x14, 0x21, 0x32, 0x46): return 0  # DRAWCTXT_DESTROY, SHAREDMEM_FREE, SETPROPERTY, GPUOBJ_FREE
    raise RuntimeError(f"unsupported kgsl ioctl {nr:#x}")

  def _device_getproperty(self, argp):
    st = kgsl.struct_kgsl_device_getproperty.from_address(argp)
    if st.type == kgsl.KGSL_PROP_DEVICE_INFO:
      info = kgsl.struct_kgsl_devinfo.from_address(int(st.value or 0))
      info.device_id, info.chip_id, info.mmu_enabled = kgsl.KGSL_DEVICE_3D0 + 1, self.gpu.gpu_id, 1
      info.gmem_gpubaseaddr, info.gmem_sizebytes = 0, 1 << 20
    return 0

  def _gpu_command(self, argp):
    st = kgsl.struct_kgsl_gpu_command.from_address(argp)
    for i in range(st.numcmds):
      obj = kgsl.struct_kgsl_command_object.from_address(st.cmdlist + i * st.cmdsize)
      if obj.flags & kgsl.KGSL_CMDLIST_IB: self.gpu.execute(obj.gpuaddr, obj.size // 4)
    self.timestamp += 1
    st.timestamp = self.timestamp
    return 0
