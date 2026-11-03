; printable_area = 0x0,100x0,100x100,0x100
M83
G1 X10 Y10 Z.2
;TYPE:Outer wall
; start printing object, unique label id: 14
M624 AQAAAAAAAAA=
G1 X20 E1 F600
G1 Y11
G1 X10 E1
; stop printing object, unique label id: 14
M625
G1 X30 Y10
; start printing object, unique label id: 25
M624 AgAAAAAAAAA=
G1 X40 E1
G1 Y11
G1 X30 E1
; stop printing object, unique label id: 25
M625
G1 X0 Y0
