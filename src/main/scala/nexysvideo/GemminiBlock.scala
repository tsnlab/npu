
import chisel3._
import chisel3.util._
import gemmini._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.rocket.MStatus

class GemminiBlock(implicit p: Parameters)  {
  override lazy val module = new GemminiBlockImp(this)
  val tlNode = module.id_node
  val atlNode = TLIdentityNode()
}

class GemminiBlockImp(outer: Example)(implicit p: Parameters) extends LazyModuleImp(outer) {

  val io = new Bundle {
    val rcc_valid = Input(Bool())
    val rcc_ready = Output(Bool())
    val rcc_dram_addr = Input(UInt(40.W))
    val rcc_dpram_addr = Input(UInt(16.W))
    val rcc_length = Input(UInt(16.W))
    val rcd_valid = Output(Bool())
    val rcd_ready = Input(Bool())
    val rcd_dpram_addr = Output(UInt(16.W))
    val rcd_read_data = Output(UInt(128.W))
    val rcd_length = Output(UInt(16.W))
    val wcc_write_valid = Input(Bool())
    val wcc_write_ready = Output(Bool())
    val wcc_write_data = Input(UInt(128.W))
    val wcc_write_length = Input(UInt(16.W))
    val wcc_dram_addr = Input(UInt(40.W))
    val mstatus = new MStatus
    val ptw = new TLBPTWIO
  }

  val max_in_flight_mem_reqs = 16
  val dataBits = 128 // dma_buswidth
  val maxBytes = 64 // dma_maxbytes

  val dummyConfig = GemminiArrayConfig[DummySInt, Float, Float](
    inputType = DummySInt(128),
    accType = DummySInt(128),
    spatialArrayOutputType = DummySInt(128),
    tileRows     = 1,
    tileColumns  = 1,
    meshRows     = 1,
    meshColumns  = 1,
    dataflow     = defaultConfig.dataflow,
    sp_capacity  = CapacityInKilobytes(128),
    acc_capacity = CapacityInKilobytes(128),
    sp_banks     = 1,
    acc_banks    = 1,
    sp_singleported = defaultConfig.sp_singleported,
    acc_singleported = defaultConfig.acc_singleported,
    has_training_convs = false,
    has_max_pool = false,
    has_nonlinear_activations = false,
    reservation_station_entries_ld = defaultConfig.reservation_station_entries_ld,
    reservation_station_entries_st = defaultConfig.reservation_station_entries_st,
    reservation_station_entries_ex = defaultConfig.reservation_station_entries_ex,
    ld_queue_length = defaultConfig.ld_queue_length,
    st_queue_length = defaultConfig.st_queue_length,
    ex_queue_length = defaultConfig.ex_queue_length,
    max_in_flight_mem_reqs = 16,
    dma_maxbytes = 64,
    dma_buswidth = 128,
    tlb_size = defaultConfig.tlb_size,

    mvin_scale_args = Some(ScaleArguments(
      (t: DummySInt, f: SInt) => t.dontCare,
      4, SInt(128), 4,
      identity = "1.0",
      c_str = "()"
    )),

    mvin_scale_acc_args = None,
    mvin_scale_shared = defaultConfig.mvin_scale_shared,

    acc_scale_args = Some(ScaleArguments(
      (t: DummySInt, f: Float) => t.dontCare,
      1, SInt(128), -1,
      identity = "1",
      c_str = "()"
    )),

    num_counter = 0,

    acc_read_full_width = false,
    acc_read_small_width = defaultConfig.acc_read_small_width,

    ex_read_from_spad = defaultConfig.ex_read_from_spad,
    ex_read_from_acc = false,
    ex_write_to_spad = false,
    ex_write_to_acc = defaultConfig.ex_write_to_acc,
  )

  val spad_w = 128
  val acc_w = 128
  val aligned_to = 1
  val sp_bank_entries = 128
  val acc_bank_entries = 128
  val block_rows = 1
  val block_cols = 1
  val use_tlb_register_filter = true
  val use_firesim_simulation_counters = false
  val use_shared_tlb = true

  val reader = LazyModule(new StreamReader(config, max_in_flight_mem_reqs, dataBits, maxBytes, spad_w, acc_w, aligned_to,
    sp_bank_entries, acc_bank_entries, block_rows, use_tlb_register_filter,
    use_firesim_simulation_counters))
  val writer = LazyModule(new StreamWriter(max_in_flight_mem_reqs, dataBits, maxBytes,
    spad_w, aligned_to, SInt(128), block_cols, use_tlb_register_filter,
    use_firesim_simulation_counters))

  val id_node = TLIdentityNode()
  val xbar_node = TLXbar()
  xbar_node := TLBuffer() := reader.node
  xbar_node := TLBuffer() := writer.node
  id_node := TLWidthWidget(16) := TLBuffer() := xbar_node

    reader.module.io.req.valid := io.rcc_valid
    io.rcc_ready := reader.module.io.req.ready
    reader.module.io.req.bits.vaddr := io.rcc_dram_addr
    reader.module.io.req.bits.spaddr := io.rcc_dpram_addr
    reader.module.io.req.bits.len := io.rcc_length
    reader.module.io.req.bits.repeats := 0.U
    reader.module.io.req.bits.pixel_repeats := 0.U
    reader.module.io.req.bits.scale := 1.U
    reader.module.io.req.bits.is_acc := false.B
    reader.module.io.req.bits.accumulate := false.B
    reader.module.io.req.bits.has_acc_bitwidth := false.B
    reader.module.io.req.bits.block_stride := 1.U
    reader.module.io.req.bits.status := io.mstatus
    reader.module.io.req.bits.cmd_id := 0.U

    io.rcd_valid := reader.module.io.resp.valid
    reader.module.io.resp.ready := io.rcd_ready
    io.rcd_read_data := reader.module.io.resp.bits.data
    io.rcd_dpram_addr := reader.module.io.resp.bits.addr
    io.rcd_length := reader.module.io.resp.bits.len

    writer.module.io.req.valid := io.wcc_write_valid
    io.wcc_write_ready := writer.module.io.req.ready
    writer.module.io.req.bits.vaddr := io.wcc_dram_addr
    writer.module.io.req.bits.len := io.wcc_write_length
    writer.module.io.req.bits.data := io.wcc_write_data
    writer.module.io.req.bits.block := 16.U
    writer.module.io.req.bits.status := io.mstatus
    writer.module.io.req.bits.pool_en := false.B 
    writer.module.io.req.bits.store_en := false.B

  val tlb = Module(new FrontendTLB(2, tlb_size, dma_maxbytes, use_tlb_register_filter, use_firesim_simulation_counters, use_shared_tlb))
  tlb.io.clients(0) <> reader.module.io.tlb
  tlb.io.clients(1) <> writer.module.io.tlb

  io.ptw <> tlb.io.ptw

  reader.module.io.flush := false.B
  reader.module.io.counter.external_reset := false.B
  writer.module.io.flush := false.B
  writer.module.io.counter.external_reset := false.B
}

// new chipyard.config.WithSystemBusWidth(128) ++