* Reduced single-cell SRAM read case with medium partial peripherals
.include "surrogate_nfet_freepdk45.spice"
.include "surrogate_pfet_freepdk45.spice"
.include "params.sp"

VVDD vdd vss 1.0
VVSS vss 0 0
VPREB pre_b vss PULSE(0 1.0 0.35n 20p 20p 5n 10n)
VWL_IN wl_in vss PULSE(0 1.0 0.7n 20p 20p 3.8n 8n)
CBL bl vss 4.0e-14
CBLB blb vss 4.0e-14

Xmos_0 bl wl q vss surrogate_nfet_freepdk45 l=5e-08 w=1.35e-07 scale_vth0={M0_vth0} scale_u0={M0_u0} scale_voff={M0_voff} * cell.pg_left
Xmos_1 blb wl qb vss surrogate_nfet_freepdk45 l=5e-08 w=1.35e-07 scale_vth0={M1_vth0} scale_u0={M1_u0} scale_voff={M1_voff} * cell.pg_right
Xmos_2 q qb vss vss surrogate_nfet_freepdk45 l=5e-08 w=2.05e-07 scale_vth0={M2_vth0} scale_u0={M2_u0} scale_voff={M2_voff} * cell.pd_left
Xmos_3 qb q vss vss surrogate_nfet_freepdk45 l=5e-08 w=2.05e-07 scale_vth0={M3_vth0} scale_u0={M3_u0} scale_voff={M3_voff} * cell.pd_right
Xmos_4 q qb vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M4_vth0} scale_u0={M4_u0} scale_voff={M4_voff} * cell.pu_left
Xmos_5 qb q vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M5_vth0} scale_u0={M5_u0} scale_voff={M5_voff} * cell.pu_right
Xmos_6 bl pre_b vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M6_vth0} scale_u0={M6_u0} scale_voff={M6_voff} * precharge.bl
Xmos_7 blb pre_b vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M7_vth0} scale_u0={M7_u0} scale_voff={M7_voff} * precharge.blb
Xmos_8 bl pre_b blb vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M8_vth0} scale_u0={M8_u0} scale_voff={M8_voff} * precharge.eq
Xmos_9 ctrl0 pre_b vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M9_vth0} scale_u0={M9_u0} scale_voff={M9_voff} * local_ctrl.inv0.mp
Xmos_10 ctrl0 pre_b vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M10_vth0} scale_u0={M10_u0} scale_voff={M10_voff} * local_ctrl.inv0.mn
Xmos_11 ctrl1 ctrl0 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M11_vth0} scale_u0={M11_u0} scale_voff={M11_voff} * local_ctrl.inv1.mp
Xmos_12 ctrl1 ctrl0 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M12_vth0} scale_u0={M12_u0} scale_voff={M12_voff} * local_ctrl.inv1.mn
Xmos_13 ctrl2 ctrl1 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M13_vth0} scale_u0={M13_u0} scale_voff={M13_voff} * local_ctrl.inv2.mp
Xmos_14 ctrl2 ctrl1 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M14_vth0} scale_u0={M14_u0} scale_voff={M14_voff} * local_ctrl.inv2.mn
Xmos_15 ctrl3 ctrl2 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M15_vth0} scale_u0={M15_u0} scale_voff={M15_voff} * local_ctrl.inv3.mp
Xmos_16 ctrl3 ctrl2 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M16_vth0} scale_u0={M16_u0} scale_voff={M16_voff} * local_ctrl.inv3.mn
Xmos_17 wl_d1 wl_in vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M17_vth0} scale_u0={M17_u0} scale_voff={M17_voff} * wl_driver.inv0.mp
Xmos_18 wl_d1 wl_in vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M18_vth0} scale_u0={M18_u0} scale_voff={M18_voff} * wl_driver.inv0.mn
Xmos_19 wl_d2 wl_d1 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=5.4e-07 scale_vth0={M19_vth0} scale_u0={M19_u0} scale_voff={M19_voff} * wl_driver.inv1.mp
Xmos_20 wl_d2 wl_d1 vss vss surrogate_nfet_freepdk45 l=5e-08 w=1.8e-07 scale_vth0={M20_vth0} scale_u0={M20_u0} scale_voff={M20_voff} * wl_driver.inv1.mn
Xmos_21 wl_d3 wl_d2 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=1.08e-06 scale_vth0={M21_vth0} scale_u0={M21_u0} scale_voff={M21_voff} * wl_driver.inv2.mp
Xmos_22 wl_d3 wl_d2 vss vss surrogate_nfet_freepdk45 l=5e-08 w=3.6e-07 scale_vth0={M22_vth0} scale_u0={M22_u0} scale_voff={M22_voff} * wl_driver.inv2.mn
Xmos_23 wl wl_d3 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.16e-06 scale_vth0={M23_vth0} scale_u0={M23_u0} scale_voff={M23_voff} * wl_driver.inv3.mp
Xmos_24 wl wl_d3 vss vss surrogate_nfet_freepdk45 l=5e-08 w=7.2e-07 scale_vth0={M24_vth0} scale_u0={M24_u0} scale_voff={M24_voff} * wl_driver.inv3.mn
Vdummy0 dummy_in0 vss 0
Xmos_25 dummy_out0 dummy_in0 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M25_vth0} scale_u0={M25_u0} scale_voff={M25_voff} * dummy_load0.a.mp
Xmos_26 dummy_out0 dummy_in0 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M26_vth0} scale_u0={M26_u0} scale_voff={M26_voff} * dummy_load0.a.mn
Xmos_27 dummy_bl0 bl vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M27_vth0} scale_u0={M27_u0} scale_voff={M27_voff} * dummy_load0.b.mp
Xmos_28 dummy_bl0 bl vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M28_vth0} scale_u0={M28_u0} scale_voff={M28_voff} * dummy_load0.b.mn
Vdummy1 dummy_in1 vss 0
Xmos_29 dummy_out1 dummy_in1 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M29_vth0} scale_u0={M29_u0} scale_voff={M29_voff} * dummy_load1.a.mp
Xmos_30 dummy_out1 dummy_in1 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M30_vth0} scale_u0={M30_u0} scale_voff={M30_voff} * dummy_load1.a.mn
Xmos_31 dummy_bl1 bl vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M31_vth0} scale_u0={M31_u0} scale_voff={M31_voff} * dummy_load1.b.mp
Xmos_32 dummy_bl1 bl vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M32_vth0} scale_u0={M32_u0} scale_voff={M32_voff} * dummy_load1.b.mn
Vdummy2 dummy_in2 vss 0
Xmos_33 dummy_out2 dummy_in2 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M33_vth0} scale_u0={M33_u0} scale_voff={M33_voff} * dummy_load2.a.mp
Xmos_34 dummy_out2 dummy_in2 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M34_vth0} scale_u0={M34_u0} scale_voff={M34_voff} * dummy_load2.a.mn
Xmos_35 dummy_bl2 bl vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M35_vth0} scale_u0={M35_u0} scale_voff={M35_voff} * dummy_load2.b.mp
Xmos_36 dummy_bl2 bl vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M36_vth0} scale_u0={M36_u0} scale_voff={M36_voff} * dummy_load2.b.mn
Vdummy3 dummy_in3 vss 0
Xmos_37 dummy_out3 dummy_in3 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M37_vth0} scale_u0={M37_u0} scale_voff={M37_voff} * dummy_load3.a.mp
Xmos_38 dummy_out3 dummy_in3 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M38_vth0} scale_u0={M38_u0} scale_voff={M38_voff} * dummy_load3.a.mn
Xmos_39 dummy_bl3 bl vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M39_vth0} scale_u0={M39_u0} scale_voff={M39_voff} * dummy_load3.b.mp
Xmos_40 dummy_bl3 bl vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M40_vth0} scale_u0={M40_u0} scale_voff={M40_voff} * dummy_load3.b.mn
Vdummy4 dummy_in4 vss 0
Xmos_41 dummy_out4 dummy_in4 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M41_vth0} scale_u0={M41_u0} scale_voff={M41_voff} * dummy_load4.a.mp
Xmos_42 dummy_out4 dummy_in4 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M42_vth0} scale_u0={M42_u0} scale_voff={M42_voff} * dummy_load4.a.mn
Xmos_43 dummy_bl4 bl vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M43_vth0} scale_u0={M43_u0} scale_voff={M43_voff} * dummy_load4.b.mp
Xmos_44 dummy_bl4 bl vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M44_vth0} scale_u0={M44_u0} scale_voff={M44_voff} * dummy_load4.b.mn
Vdummy5 dummy_in5 vss 0
Xmos_45 dummy_out5 dummy_in5 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M45_vth0} scale_u0={M45_u0} scale_voff={M45_voff} * dummy_load5.a.mp
Xmos_46 dummy_out5 dummy_in5 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M46_vth0} scale_u0={M46_u0} scale_voff={M46_voff} * dummy_load5.a.mn
Xmos_47 dummy_bl5 bl vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M47_vth0} scale_u0={M47_u0} scale_voff={M47_voff} * dummy_load5.b.mp
Xmos_48 dummy_bl5 bl vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M48_vth0} scale_u0={M48_u0} scale_voff={M48_voff} * dummy_load5.b.mn
Xmos_49 ro0 bl vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M49_vth0} scale_u0={M49_u0} scale_voff={M49_voff} * readout.inv0.mp
Xmos_50 ro0 bl vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M50_vth0} scale_u0={M50_u0} scale_voff={M50_voff} * readout.inv0.mn
Xmos_51 ro1 ro0 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M51_vth0} scale_u0={M51_u0} scale_voff={M51_voff} * readout.inv1.mp
Xmos_52 ro1 ro0 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M52_vth0} scale_u0={M52_u0} scale_voff={M52_voff} * readout.inv1.mn
Xmos_53 ro2 ro1 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M53_vth0} scale_u0={M53_u0} scale_voff={M53_voff} * readout.inv2.mp
Xmos_54 ro2 ro1 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M54_vth0} scale_u0={M54_u0} scale_voff={M54_voff} * readout.inv2.mn
Xmos_55 ro3 ro2 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M55_vth0} scale_u0={M55_u0} scale_voff={M55_voff} * readout.inv3.mp
Xmos_56 ro3 ro2 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M56_vth0} scale_u0={M56_u0} scale_voff={M56_voff} * readout.inv3.mn
Xmos_57 ro4 ro3 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M57_vth0} scale_u0={M57_u0} scale_voff={M57_voff} * readout.inv4.mp
Xmos_58 ro4 ro3 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M58_vth0} scale_u0={M58_u0} scale_voff={M58_voff} * readout.inv4.mn
Xmos_59 ro5 ro4 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M59_vth0} scale_u0={M59_u0} scale_voff={M59_voff} * readout.inv5.mp
Xmos_60 ro5 ro4 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M60_vth0} scale_u0={M60_u0} scale_voff={M60_voff} * readout.inv5.mn
Xmos_61 ro6 ro5 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M61_vth0} scale_u0={M61_u0} scale_voff={M61_voff} * readout.inv6.mp
Xmos_62 ro6 ro5 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M62_vth0} scale_u0={M62_u0} scale_voff={M62_voff} * readout.inv6.mn
Xmos_63 ro7 ro6 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M63_vth0} scale_u0={M63_u0} scale_voff={M63_voff} * readout.inv7.mp
Xmos_64 ro7 ro6 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M64_vth0} scale_u0={M64_u0} scale_voff={M64_voff} * readout.inv7.mn
Xmos_65 ro8 ro7 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M65_vth0} scale_u0={M65_u0} scale_voff={M65_voff} * readout.inv8.mp
Xmos_66 ro8 ro7 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M66_vth0} scale_u0={M66_u0} scale_voff={M66_voff} * readout.inv8.mn
Xmos_67 ro9 ro8 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M67_vth0} scale_u0={M67_u0} scale_voff={M67_voff} * readout.inv9.mp
Xmos_68 ro9 ro8 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M68_vth0} scale_u0={M68_u0} scale_voff={M68_voff} * readout.inv9.mn
Xmos_69 ro10 ro9 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M69_vth0} scale_u0={M69_u0} scale_voff={M69_voff} * readout.inv10.mp
Xmos_70 ro10 ro9 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M70_vth0} scale_u0={M70_u0} scale_voff={M70_voff} * readout.inv10.mn
Xmos_71 ro11 ro10 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M71_vth0} scale_u0={M71_u0} scale_voff={M71_voff} * readout.inv11.mp
Xmos_72 ro11 ro10 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M72_vth0} scale_u0={M72_u0} scale_voff={M72_voff} * readout.inv11.mn
Xmos_73 ro12 ro11 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M73_vth0} scale_u0={M73_u0} scale_voff={M73_voff} * readout.inv12.mp
Xmos_74 ro12 ro11 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M74_vth0} scale_u0={M74_u0} scale_voff={M74_voff} * readout.inv12.mn
Xmos_75 ro13 ro12 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M75_vth0} scale_u0={M75_u0} scale_voff={M75_voff} * readout.inv13.mp
Xmos_76 ro13 ro12 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M76_vth0} scale_u0={M76_u0} scale_voff={M76_voff} * readout.inv13.mn
Xmos_77 ro14 ro13 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M77_vth0} scale_u0={M77_u0} scale_voff={M77_voff} * readout.inv14.mp
Xmos_78 ro14 ro13 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M78_vth0} scale_u0={M78_u0} scale_voff={M78_voff} * readout.inv14.mn
Xmos_79 ro15 ro14 vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=2.7e-07 scale_vth0={M79_vth0} scale_u0={M79_u0} scale_voff={M79_voff} * readout.inv15.mp
Xmos_80 ro15 ro14 vss vss surrogate_nfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M80_vth0} scale_u0={M80_u0} scale_voff={M80_voff} * readout.inv15.mn

.ic v(q)=0 v(qb)=1 v(bl)=1 v(blb)=1
.control
tran 0.005n 6.0n uic
meas tran read_error FIND v(q) AT=5.2n
quit
.endc
.end
