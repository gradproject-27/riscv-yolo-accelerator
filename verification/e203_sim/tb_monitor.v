// Extra simulation monitor for the stock E203 tb_top (compiled alongside it,
// becomes a second top-level module; it only observes, never drives).
//  - prints every byte the program writes to UART0 (printf output)
//  - stops the simulation when the program writes 0xC0DE0001 (PASS) or
//    0xC0DE0002 (FAIL) to DTCM address 0x90001000 (SIM_RESULT)

`define MON_CPU   tb_top.u_e203_soc_top.u_e203_subsys_top.u_e203_subsys_main.u_e203_cpu_top
`define MON_UART  tb_top.u_e203_soc_top.u_e203_subsys_top.u_e203_subsys_main.u_e203_subsys_perips.u_perips_apb_uart0
`define MON_DTCM  `MON_CPU.u_e203_srams.u_e203_dtcm_ram.u_e203_dtcm_gnrl_ram.u_sirv_sim_ram.mem_r

module tb_monitor();

  localparam SIM_RESULT_IDX = (32'h9000_1000 - 32'h9000_0000) >> 2;

  // UART0 TX capture (writes to THR while DLAB = 0)
  always @(posedge `MON_UART.CLK) begin
    if (`MON_UART.fifo_tx_valid && `MON_UART.fifo_tx_data != 8'h0d)
      $write("%c", `MON_UART.fifo_tx_data);
  end

  // SIM_RESULT watch
  wire [31:0] sim_result = `MON_DTCM[SIM_RESULT_IDX];
  always @(posedge tb_top.clk) begin
    if (sim_result == 32'hC0DE0001 || sim_result == 32'hC0DE0002) begin
      $display("");
      $display("==========================================");
      if (sim_result == 32'hC0DE0001)
        $display("        PROGRAM PASS");
      else
        $display("        PROGRAM FAIL");
      $display("DTCM result             = 0x%h", sim_result);
      $display("Total cycle_count       = %0d", tb_top.cycle_count);
      $display("Valid instruction count = %0d", tb_top.valid_ir_cycle);
      // Optional per-section counters a program may store after SIM_RESULT
      // (0x90001004 / 0x90001008 / 0x9000100C), e.g. conv2d_sim
      if (`MON_DTCM[SIM_RESULT_IDX+1] !== 32'bx) begin
        $display("Program-reported counters (mcycle/minstret):");
        $display("  section cycles        = %0d", `MON_DTCM[SIM_RESULT_IDX+1]);
        $display("  section instructions  = %0d", `MON_DTCM[SIM_RESULT_IDX+2]);
        $display("  extra cycles          = %0d", `MON_DTCM[SIM_RESULT_IDX+3]);
      end
      $display("==========================================");
      $finish;
    end
  end

endmodule
