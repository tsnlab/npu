#
# vivado project script for npu project
#

#---- open the graphical environment
start_gui

#---- create project & directory structure
create_project -force NPU ../NPU -part xc7z020clg400-1

#---- select zybo-z7 board
set_property board_part digilentinc.com:zybo-z7-20:part0:1.1 [current_project]

#---- add sources to the projectl
add_files -norecurse { ./src/npu.v ./src/npt.v ./src/nps.v ./src/npm.v ./src/npc.v ./src/intp.v ./src/sram.v ./src/fpu.v ./src/FDIV.v ./src/FMUL.v ./src/FADD.v }
add_files -norecurse -fileset constrs_1 ./src/npu.xdc

#---- create block design
create_bd_design "ps"

#---- add zynq7
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup
set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {125} CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_USE_S_AXI_HP1 {1} CONFIG.PCW_USE_S_AXI_HP2 {1} CONFIG.PCW_USE_S_AXI_HP3 {1} CONFIG.PCW_USE_FABRIC_INTERRUPT {1} CONFIG.PCW_IRQ_F2P_INTR {1}] [get_bd_cells processing_system7_0]

#---- add axi interconnect
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_1
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_2
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_3
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_4
endgroup
set_property -dict [list CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_0]
set_property -dict [list CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_1]
set_property -dict [list CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_2]
set_property -dict [list CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_3]
set_property -dict [list CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_4]

#---- add processor system reset
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0
endgroup

#---- run block automation
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]

#---- run connection automation
startgroup
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (125 MHz)} Freq {125} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (125 MHz)} Freq {125} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_1/ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (125 MHz)} Freq {125} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_2/ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (125 MHz)} Freq {125} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_3/ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (125 MHz)} Freq {125} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_4/ACLK]
endgroup

#---- add interrupt port
create_bd_port -dir I -from 3 -to 0 -type intr irq
set_property CONFIG.SENSITIVITY EDGE_RISING [get_bd_ports irq]

#---- add interface port(slave axi)
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 m0_axi
set_property -dict [list CONFIG.ID_WIDTH {6} CONFIG.FREQ_HZ {125000000} CONFIG.DATA_WIDTH {64}] [get_bd_intf_ports m0_axi]
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 m1_axi
set_property -dict [list CONFIG.ID_WIDTH {6} CONFIG.FREQ_HZ {125000000} CONFIG.DATA_WIDTH {64}] [get_bd_intf_ports m1_axi]
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 m2_axi
set_property -dict [list CONFIG.ID_WIDTH {6} CONFIG.FREQ_HZ {125000000} CONFIG.DATA_WIDTH {64}] [get_bd_intf_ports m2_axi]
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 m3_axi
set_property -dict [list CONFIG.ID_WIDTH {6} CONFIG.FREQ_HZ {125000000} CONFIG.DATA_WIDTH {64}] [get_bd_intf_ports m3_axi]

#---- add interface port(master axi)
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi
set_property -dict [list CONFIG.FREQ_HZ {125000000}] [get_bd_intf_ports s_axi]

#---- add clock out port
create_bd_port -dir O -type clk oclk

#---- add reset out port
create_bd_port -dir O -type rst orstn

#---- connect manually
# clock
connect_bd_net [get_bd_pins axi_interconnect_0/S00_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axi_interconnect_0/M00_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axi_interconnect_1/S00_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axi_interconnect_1/M00_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axi_interconnect_2/S00_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axi_interconnect_2/M00_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axi_interconnect_3/S00_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axi_interconnect_3/M00_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axi_interconnect_4/S00_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axi_interconnect_4/M00_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins processing_system7_0/S_AXI_HP1_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins processing_system7_0/S_AXI_HP2_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins processing_system7_0/S_AXI_HP3_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_ports oclk] [get_bd_pins processing_system7_0/FCLK_CLK0]
# reset
connect_bd_net [get_bd_pins axi_interconnect_0/S00_ARESETN] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
connect_bd_net [get_bd_pins axi_interconnect_0/M00_ARESETN] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
connect_bd_net [get_bd_pins axi_interconnect_1/S00_ARESETN] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
connect_bd_net [get_bd_pins axi_interconnect_1/M00_ARESETN] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
connect_bd_net [get_bd_pins axi_interconnect_2/S00_ARESETN] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
connect_bd_net [get_bd_pins axi_interconnect_2/M00_ARESETN] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
connect_bd_net [get_bd_pins axi_interconnect_3/S00_ARESETN] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
connect_bd_net [get_bd_pins axi_interconnect_3/M00_ARESETN] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
connect_bd_net [get_bd_pins axi_interconnect_4/S00_ARESETN] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
connect_bd_net [get_bd_pins axi_interconnect_4/M00_ARESETN] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
connect_bd_net [get_bd_ports orstn] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
# interrupt
connect_bd_net [get_bd_ports irq] [get_bd_pins processing_system7_0/IRQ_F2P]
# axi
connect_bd_intf_net [get_bd_intf_ports m0_axi] -boundary_type upper [get_bd_intf_pins axi_interconnect_1/S00_AXI]
connect_bd_intf_net [get_bd_intf_ports m1_axi] -boundary_type upper [get_bd_intf_pins axi_interconnect_2/S00_AXI]
connect_bd_intf_net [get_bd_intf_ports m2_axi] -boundary_type upper [get_bd_intf_pins axi_interconnect_3/S00_AXI]
connect_bd_intf_net [get_bd_intf_ports m3_axi] -boundary_type upper [get_bd_intf_pins axi_interconnect_4/S00_AXI]
connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_1/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_2/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP1]
connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_3/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP2]
connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_4/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP3]
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] -boundary_type upper [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_ports s_axi] -boundary_type upper [get_bd_intf_pins axi_interconnect_0/M00_AXI]

#---- assign bd address
assign_bd_address
#set_property offset 0x40000000 [get_bd_addr_segs {processing_system7_0/Data/SEG_m_axi_Reg}]

#---- validate bd design
validate_bd_design

#---- save & close bd design
save_bd_design
close_bd_design ps

#---- launch synthesis
launch_runs synth_1
wait_on_run synth_1

#---- launch implementation
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

#---- export hardware
write_hw_platform -fixed -include_bit -force -file ./npu.xsa

#---- quit
quit
