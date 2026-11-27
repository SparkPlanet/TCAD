* Reduced 6T SRAM cell-only read-disturb case
.include "surrogate_nfet_freepdk45.spice"
.include "surrogate_pfet_freepdk45.spice"
.include "params.sp"

VVDD vdd vss 1.0
VVSS vss 0 0
VBL bl vss 1.0
VBLB blb vss 1.0
VWL wl vss PULSE(0 1.0 0.2n 20p 20p 1.6n 4n)

Xmos_0 bl wl q vss surrogate_nfet_freepdk45 l=5e-08 w=1.35e-07 scale_vth0={M0_vth0} scale_u0={M0_u0} scale_voff={M0_voff} * cell.pg_left
Xmos_1 blb wl qb vss surrogate_nfet_freepdk45 l=5e-08 w=1.35e-07 scale_vth0={M1_vth0} scale_u0={M1_u0} scale_voff={M1_voff} * cell.pg_right
Xmos_2 q qb vss vss surrogate_nfet_freepdk45 l=5e-08 w=2.05e-07 scale_vth0={M2_vth0} scale_u0={M2_u0} scale_voff={M2_voff} * cell.pd_left
Xmos_3 qb q vss vss surrogate_nfet_freepdk45 l=5e-08 w=2.05e-07 scale_vth0={M3_vth0} scale_u0={M3_u0} scale_voff={M3_voff} * cell.pd_right
Xmos_4 q qb vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M4_vth0} scale_u0={M4_u0} scale_voff={M4_voff} * cell.pu_left
Xmos_5 qb q vdd vdd surrogate_pfet_freepdk45 l=5e-08 w=9e-08 scale_vth0={M5_vth0} scale_u0={M5_u0} scale_voff={M5_voff} * cell.pu_right

.ic v(q)=0 v(qb)=1
.control
tran 0.002n 2.0n uic
meas tran read_error FIND v(q) AT=1.8n
quit
.endc
.end
