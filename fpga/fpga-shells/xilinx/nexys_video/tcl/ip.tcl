# See LICENSE for license details.

create_ip -vendor xilinx.com -library ip -name clk_wiz -module_name mmcm -dir $ipdir -force
set_property -dict [list \
        CONFIG.PRIMITIVE {MMCM} \
        CONFIG.RESET_TYPE {ACTIVE_LOW} \
        CONFIG.CLKOUT1_USED {true} \
        CONFIG.CLKOUT2_USED {true} \
        CONFIG.CLKOUT3_USED {true} \
        CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {8.388} \
        CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {65.000} \
        CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {32.500} \
        ] [get_ips mmcm]

create_ip -vendor xilinx.com -library ip -name proc_sys_reset -module_name reset_sys -dir $ipdir -force
set_property -dict [list \
        CONFIG.C_EXT_RESET_HIGH {false} \
        CONFIG.C_AUX_RESET_HIGH {false} \
        CONFIG.C_NUM_BUS_RST {1} \
        CONFIG.C_NUM_PERP_RST {1} \
        CONFIG.C_NUM_INTERCONNECT_ARESETN {1} \
        CONFIG.C_NUM_PERP_ARESETN {1} \
        ] [get_ips reset_sys]

create_ip -vendor xilinx.com -library ip -name ila -module_name ila -dir $ipdir -force
set_property -dict [list \
        CONFIG.C_NUM_OF_PROBES {1} \
        CONFIG.C_TRIGOUT_EN {false} \
        CONFIG.C_TRIGIN_EN {false} \
        CONFIG.C_MONITOR_TYPE {Native} \
        CONFIG.C_ENABLE_ILA_AXI_MON {false} \
        CONFIG.C_PROBE0_WIDTH {4} \
        CONFIG.C_PROBE10_TYPE {1} \
        CONFIG.C_PROBE10_WIDTH {32} \
        CONFIG.C_PROBE11_TYPE {1} \
        CONFIG.C_PROBE11_WIDTH {32} \
        CONFIG.C_PROBE12_TYPE {1} \
        CONFIG.C_PROBE12_WIDTH {64} \
        CONFIG.C_PROBE13_TYPE {1} \
        CONFIG.C_PROBE13_WIDTH {64} \
        CONFIG.C_PROBE14_TYPE {1} \
        CONFIG.C_PROBE14_WIDTH {97} \
        ] [get_ips ila]

create_ip -vendor xilinx.com -library ip -name fifo_generator -module_name as32x512_ft -dir $ipdir -force
set_property -dict [list \
        CONFIG.Fifo_Implementation {Independent_Clocks_Distributed_RAM} \
        CONFIG.Performance_Options {First_Word_Fall_Through} \
        CONFIG.Input_Data_Width {32} \
        CONFIG.Input_Depth {512} \
        CONFIG.Output_Data_Width {32} \
        CONFIG.Output_Depth {512} \
        CONFIG.Reset_Type {Asynchronous_Reset} \
        CONFIG.Full_Flags_Reset_Value {1} \
        CONFIG.Data_Count_Width {9} \
        CONFIG.Write_Data_Count {true} \
        CONFIG.Write_Data_Count_Width {9} \
        CONFIG.Read_Data_Count {true} \
        CONFIG.Read_Data_Count_Width {9} \
        CONFIG.Full_Threshold_Assert_Value {511} \
        CONFIG.Full_Threshold_Negate_Value {510} \
        CONFIG.Empty_Threshold_Assert_Value {4} \
        CONFIG.Empty_Threshold_Negate_Value {5}
        ] [get_ips as32x512_ft]

create_ip -name fifo_generator -vendor xilinx.com -library ip -module_name as72x512_ft -dir $ipdir -force
set_property -dict [list \
        CONFIG.Fifo_Implementation {Independent_Clocks_Distributed_RAM} \
        CONFIG.Performance_Options {First_Word_Fall_Through} \
        CONFIG.Input_Data_Width {72} \
        CONFIG.Input_Depth {512} \
        CONFIG.Output_Data_Width {72} \
        CONFIG.Output_Depth {512} \
        CONFIG.Reset_Type {Asynchronous_Reset} \
        CONFIG.Full_Flags_Reset_Value {1} \
        CONFIG.Data_Count_Width {9} \
        CONFIG.Write_Data_Count {true} \
        CONFIG.Write_Data_Count_Width {9} \
        CONFIG.Read_Data_Count {true} \
        CONFIG.Read_Data_Count_Width {9} \
        CONFIG.Full_Threshold_Assert_Value {511} \
        CONFIG.Full_Threshold_Negate_Value {510} \
        CONFIG.Empty_Threshold_Assert_Value {4} \
        CONFIG.Empty_Threshold_Negate_Value {5}
        ] [get_ips as72x512_ft]

create_ip -name fifo_generator -vendor xilinx.com -library ip -module_name as128x1024 -dir $ipdir -force
set_property -dict [list \
        CONFIG.Fifo_Implementation {Independent_Clocks_Distributed_RAM} \
        CONFIG.INTERFACE_TYPE {Native}\
        CONFIG.Performance_Options {First_Word_Fall_Through} \
        CONFIG.Input_Data_Width {128} \
        CONFIG.Output_Data_Width {128} \
        CONFIG.Reset_Type {Asynchronous_Reset} \
        CONFIG.Full_Flags_Reset_Value {1} \
        CONFIG.Write_Data_Count {true}\
        CONFIG.Read_Data_Count {true} \
        CONFIG.Full_Threshold_Assert_Value {1023} \
        CONFIG.Full_Threshold_Negate_Value {1022} \
        CONFIG.Empty_Threshold_Assert_Value {4} \
        CONFIG.Empty_Threshold_Negate_Value {5} \
        CONFIG.FIFO_Implementation_wach {Common_Clock_Distributed_RAM} \
        CONFIG.Full_Threshold_Assert_Value_wach {15} \
        CONFIG.Empty_Threshold_Assert_Value_wach {14} \
        CONFIG.FIFO_Implementation_wrch {Common_Clock_Distributed_RAM} \
        CONFIG.Full_Threshold_Assert_Value_wrch {15} \
        CONFIG.Empty_Threshold_Assert_Value_wrch {14} \
        CONFIG.FIFO_Implementation_rach {Common_Clock_Distributed_RAM} \
        CONFIG.Full_Threshold_Assert_Value_rach {15} \
        CONFIG.Empty_Threshold_Assert_Value_rach {14} \
        CONFIG.Enable_Safety_Circuit {false}
        ] [get_ips as128x1024]

