@echo off
setlocal

rem 定义参数列表
@REM set params="Ukulele_01" "antique_katana_01" "boulder_01" "chinese_chandelier" "dartboard" "Drill_01" "dry_branches_medium_01" "food_kiwi_01" "hand_plane_no4" "power_box_01" "sofa_03" "Television_01"
set params="lambis_shell" "lubricant_spray" "treasure_chest" "metal_trash_can"
set run_mode="2"
set refinecount="0"
set epoch="4000"
set lr="0.01"
set quantizeMode="4"
set optimizeMode="1"
set mode7Type="1"
set pretain="0"

@REM train
@REM (for %%p in (%params%) do (
@REM     call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% "1" %mode7Type% "1" "0" %%p
@REM     call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% "0" %mode7Type% "0" "0" %%p
@REM ))

@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "Ukulele_01" "600" "1600"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "antique_katana_01" "700" "1500"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "boulder_01" "500" "1550"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "chinese_chandelier" "300" "1500"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "dartboard" "700" "1500"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "Drill_01" "600" "1600"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "dry_branches_medium_01" "450" "1550"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "food_kiwi_01" "50" "1500"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "hand_plane_no4" "400" "1550"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "power_box_01" "400" "1600"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "sofa_03" "300" "1600"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "Television_01" "200" "1700"

call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "lambis_shell" "350" "1550"
call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "lubricant_spray" "300" "1600"
call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "treasure_chest" "550" "1500"
call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %quantizeMode% %optimizeMode% %mode7Type% "0" "1" "metal_trash_can" "450" "1650"

endlocal